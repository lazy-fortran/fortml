module fortml_gp_variational_categorical_classification
    !! Coupled categorical variational GP classification.
    !!
    !! The model owns one inducing posterior per sorted class, but unlike the
    !! one-vs-rest adapter it uses one shared categorical likelihood.  The
    !! corrected latent logits are
    !! ``mu/sqrt(1+c*variance)`` (``c=pi/8`` for logistic and ``c=1`` for
    !! probit), and a stable row-wise softmax couples all classes.  The
    !! objective is the deterministic categorical ELBO (weighted log-softmax
    !! likelihood minus the sum of analytic inducing KL terms).  This is a
    !! deliberately bounded CPU reference: all products are analytic, while
    !! CUDA returns a typed refusal until resident inducing solves and softmax
    !! reductions are linked.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_kernels, only: kernel_t
    use fortml_gp_variational_classification, only: &
        gp_variational_classification_t, GP_VARIATIONAL_LOGISTIC, &
        GP_VARIATIONAL_PROBIT
    use fortopt_objective, only: objective_t
    use fortopt_lbfgsb, only: lbfgsb_t, lbfgsb_options_t, lbfgsb_result_t
    implicit none
    private

    real(dp), parameter :: PI = 3.1415926535897932384626433832795_dp

    type, public :: gp_variational_categorical_options_t
        !! Initialization and bounded FortOpt controls for `fit`.
        integer :: likelihood = GP_VARIATIONAL_LOGISTIC
        integer :: n_mc_samples = 16
        integer :: seed = 1
        real(dp) :: jitter = 1.0e-8_dp
        integer :: memory = 10
        integer :: max_iterations = 100
        integer :: max_line_search = 40
        real(dp) :: gradient_tolerance = 1.0e-6_dp
        real(dp) :: step_tolerance = 1.0e-12_dp
        real(dp) :: objective_tolerance = 1.0e-12_dp
        real(dp) :: lower_bound = -20.0_dp
        real(dp) :: upper_bound = 20.0_dp
    end type gp_variational_categorical_options_t

    type, public :: gp_variational_categorical_state_t
        !! Diagnostics returned by `fit`.
        logical :: converged = .false.
        integer :: iterations = 0
        integer :: line_search_evaluations = 0
        real(dp) :: elbo = -huge(1.0_dp)
        real(dp) :: gradient_norm = huge(1.0_dp)
    end type gp_variational_categorical_state_t

    type, public :: gp_variational_categorical_classification_t
        private
        type(gp_variational_classification_t), allocatable :: models(:)
        integer, allocatable :: class_label(:)
        integer :: n_classes = 0
        integer :: n_features = 0
        integer :: likelihood = GP_VARIATIONAL_LOGISTIC
        logical :: is_initialized = .false.
    contains
        procedure, public :: initialize => gvcc_initialize
        procedure, public :: fit => gvcc_fit
        procedure, public :: set_parameters => gvcc_set_parameters
        procedure, public :: parameters => gvcc_parameters
        procedure, public :: parameter_count => gvcc_parameter_count
        procedure, public :: elbo => gvcc_elbo
        procedure, public :: elbo_gradient => gvcc_elbo_gradient
        procedure, public :: elbo_jvp => gvcc_elbo_jvp
        procedure, public :: predict_latent => gvcc_predict_latent
        procedure, public :: predict_proba => gvcc_predict_proba
        procedure, public :: predict => gvcc_predict
        procedure, public :: predict_proba_parameter_jvp => &
            gvcc_predict_proba_parameter_jvp
        procedure, public :: predict_proba_parameter_vjp => &
            gvcc_predict_proba_parameter_vjp
        procedure, public :: predict_proba_input_jvp => &
            gvcc_predict_proba_input_jvp
        procedure, public :: predict_proba_input_vjp => &
            gvcc_predict_proba_input_vjp
        procedure, public :: predict_proba_device => gvcc_predict_proba_device
        procedure, public :: predict_proba_parameter_vjp_device => &
            gvcc_predict_proba_parameter_vjp_device
        procedure, public :: predict_proba_input_vjp_device => &
            gvcc_predict_proba_input_vjp_device
        procedure, public :: elbo_device => gvcc_elbo_device
        procedure, public :: classes => gvcc_classes
        procedure, public :: class_count => gvcc_class_count
        procedure, public :: feature_count => gvcc_feature_count
        procedure, public :: initialized => gvcc_initialized
        procedure, public :: device_supported => gvcc_device_supported
    end type gp_variational_categorical_classification_t

    type :: categorical_context_t
        type(gp_variational_categorical_classification_t), pointer :: model => null()
        real(dp), allocatable :: x(:, :)
        integer, allocatable :: labels(:)
        real(dp), allocatable :: sample_weight(:)
    end type categorical_context_t

contains

    subroutine gvcc_initialize(self, inducing_points, classes, kernel, n_mc_samples, &
            seed, status, likelihood, jitter)
        class(gp_variational_categorical_classification_t), intent(out) :: self
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
        if (size(classes) < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: at least two classes are required")
            return
        end if
        if (size(inducing_points, 1) < 1 .or. size(inducing_points, 2) /= kernel%input_dim) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: inducing-point shape is invalid")
            return
        end if
        if (n_mc_samples < 1 .or. requested_jitter <= 0.0_dp .or. &
                .not. ieee_is_finite(requested_jitter)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: initialization options are invalid")
            return
        end if
        if (requested_likelihood /= GP_VARIATIONAL_LOGISTIC .and. &
                requested_likelihood /= GP_VARIATIONAL_PROBIT) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: likelihood is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(inducing_points))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: inducing points are not finite")
            return
        end if
        call sorted_unique(classes, sorted_classes)
        if (size(sorted_classes) /= size(classes)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: class labels must be unique")
            return
        end if
        self%n_classes = size(sorted_classes)
        self%n_features = kernel%input_dim
        self%likelihood = requested_likelihood
        allocate(self%class_label(self%n_classes), self%models(self%n_classes))
        self%class_label = sorted_classes
        do i = 1, self%n_classes
            call self%models(i)%initialize(inducing_points, kernel, n_mc_samples, &
                seed + i - 1, status, likelihood=requested_likelihood, jitter=requested_jitter)
            if (status%code /= FORTNUM_OK) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "categorical variational GP: class initialization failed")
                return
            end if
        end do
        self%is_initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvcc_initialize

    subroutine gvcc_fit(self, x, labels, inducing_points, kernel, status, options, &
            state, sample_weight)
        class(gp_variational_categorical_classification_t), intent(out), target :: self
        real(dp), intent(in) :: x(:, :), inducing_points(:, :)
        integer, intent(in) :: labels(:)
        type(kernel_t), intent(in) :: kernel
        type(fortnum_status_t), intent(out) :: status
        type(gp_variational_categorical_options_t), intent(in), optional :: options
        type(gp_variational_categorical_state_t), intent(out), optional :: state
        real(dp), intent(in), optional :: sample_weight(:)
        type(gp_variational_categorical_options_t) :: requested
        type(categorical_context_t), target :: context
        type(objective_t) :: objective
        type(lbfgsb_t) :: optimizer
        type(lbfgsb_options_t) :: optimizer_options
        type(lbfgsb_result_t) :: optimizer_result
        type(gp_variational_categorical_state_t) :: result
        type(fortnum_status_t) :: restore_status
        real(dp), allocatable :: parameters(:), initial_parameters(:), lower(:), upper(:), gradient(:)
        integer :: n_parameters
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(gp_variational_categorical_options_t) :: gp_variational_categorical_options_t_default
        type(gp_variational_categorical_state_t) :: gp_variational_categorical_state_t_default

        result = gp_variational_categorical_state_t_default
        if (present(state)) state = result
        requested = gp_variational_categorical_options_t_default
        if (present(options)) requested = options
        if (.not. valid_options(requested)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP fit: options are invalid")
            return
        end if
        if (.not. valid_training_data(x, labels, sample_weight)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP fit: data or labels are invalid")
            return
        end if
        call self%initialize(inducing_points, unique_labels_of(labels), kernel, &
            requested%n_mc_samples, requested%seed, status, &
            likelihood=requested%likelihood, jitter=requested%jitter)
        if (status%code /= FORTNUM_OK) return
        n_parameters = self%parameter_count()
        context%model => self
        allocate(context%x, source=x)
        allocate(context%labels, source=labels)
        if (present(sample_weight)) allocate(context%sample_weight, source=sample_weight)
        parameters = self%parameters()
        initial_parameters = parameters
        allocate(lower(n_parameters), upper(n_parameters), gradient(n_parameters))
        lower = requested%lower_bound
        upper = requested%upper_bound
        call objective%initialize_context(n_parameters, context, categorical_objective, status)
        if (status%code /= FORTNUM_OK) return
        call copy_lbfgsb_options(requested, optimizer_options)
        call optimizer%minimize(objective, parameters, lower, upper, optimizer_options, &
            optimizer_result, status)
        if (status%code /= FORTNUM_OK .and. status%code /= FORTNUM_CONVERGENCE_ERROR) then
            call self%set_parameters(initial_parameters, restore_status)
            return
        end if
        call categorical_objective(context, parameters, result%elbo, gradient, status)
        if (status%code /= FORTNUM_OK) then
            call self%set_parameters(initial_parameters, restore_status)
            return
        end if
        result%elbo = -result%elbo
        result%gradient_norm = sqrt(sum(gradient*gradient))
        result%converged = optimizer_result%state%converged
        result%iterations = optimizer_result%state%iteration
        result%line_search_evaluations = optimizer_result%line_search_evaluations
        if (.not. ieee_is_finite(result%elbo) .or. &
                .not. ieee_is_finite(result%gradient_norm)) then
            call self%set_parameters(initial_parameters, restore_status)
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "categorical variational GP fit: result is not finite")
            return
        end if
        if (.not. result%converged) then
            call self%set_parameters(initial_parameters, restore_status)
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "categorical variational GP fit: iteration limit reached")
            return
        end if
        call self%set_parameters(parameters, status)
        if (present(state)) state = result
        if (status%code /= FORTNUM_OK) return
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvcc_fit

    integer function gvcc_parameter_count(self) result(count)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        integer :: i

        count = 0
        if (.not. self%initialized()) return
        do i = 1, self%n_classes
            count = count + self%models(i)%parameter_count()
        end do
    end function gvcc_parameter_count

    function gvcc_parameters(self) result(parameters)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), allocatable :: parameters(:), local(:)
        integer :: i, first, last

        allocate(parameters(self%parameter_count()))
        if (.not. self%initialized()) return
        first = 1
        do i = 1, self%n_classes
            local = self%models(i)%parameters()
            last = first + size(local) - 1
            parameters(first:last) = local
            first = last + 1
        end do
    end function gvcc_parameters

    subroutine gvcc_set_parameters(self, parameters, status)
        class(gp_variational_categorical_classification_t), intent(inout) :: self
        real(dp), intent(in) :: parameters(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, first, last, local_count

        if (.not. self%initialized()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: model is not initialized")
            return
        end if
        if (size(parameters) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: packed parameter shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(parameters))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: packed parameters are not finite")
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
    end subroutine gvcc_set_parameters

    subroutine gvcc_elbo(self, x, labels, value, status, scale, sample_weight)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: scale
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: means(:, :), variances(:, :), logits(:, :), probabilities(:, :)
        real(dp), allocatable :: weights(:)
        real(dp) :: likelihood, divergence, multiplier
        integer :: i, j, class_index

        value = 0.0_dp
        if (.not. valid_data(self, x, labels, status, sample_weight)) return
        multiplier = 1.0_dp
        if (present(scale)) multiplier = scale
        if (multiplier <= 0.0_dp .or. .not. ieee_is_finite(multiplier)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: likelihood scale is invalid")
            return
        end if
        allocate(means(size(x, 1), self%n_classes), variances(size(x, 1), self%n_classes), &
            logits(size(x, 1), self%n_classes), probabilities(size(x, 1), self%n_classes))
        call self%predict_latent(x, means, variances, status)
        if (status%code /= FORTNUM_OK) return
        call logits_and_probabilities(self, means, variances, logits, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call observation_weights(size(labels), sample_weight, weights, status)
        if (status%code /= FORTNUM_OK) return
        likelihood = 0.0_dp
        do i = 1, size(labels)
            class_index = find_class(self%class_label, labels(i))
            likelihood = likelihood + weights(i)*log(max(probabilities(i, class_index), tiny(1.0_dp)))
        end do
        divergence = 0.0_dp
        do j = 1, self%n_classes
            call self%models(j)%kl(value, status)
            if (status%code /= FORTNUM_OK) return
            divergence = divergence + value
        end do
        value = multiplier*likelihood - divergence
        if (.not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: ELBO is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvcc_elbo

    subroutine gvcc_elbo_gradient(self, x, labels, value, gradient, status, scale, &
            sample_weight)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: scale
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: means(:, :), variances(:, :), logits(:, :), probabilities(:, :)
        real(dp), allocatable :: mean_bar(:), variance_bar(:), weights(:), local_gradient(:)
        real(dp), allocatable :: kl_gradient(:)
        real(dp) :: multiplier, likelihood, divergence, term, c, scale_value
        integer :: i, j, class_index, first, last, local_count

        value = 0.0_dp
        gradient = 0.0_dp
        if (.not. valid_data(self, x, labels, status, sample_weight)) return
        if (size(gradient) /= self%parameter_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: gradient shape is invalid")
            return
        end if
        multiplier = 1.0_dp
        if (present(scale)) multiplier = scale
        if (multiplier <= 0.0_dp .or. .not. ieee_is_finite(multiplier)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: likelihood scale is invalid")
            return
        end if
        allocate(means(size(x, 1), self%n_classes), variances(size(x, 1), self%n_classes), &
            logits(size(x, 1), self%n_classes), probabilities(size(x, 1), self%n_classes), &
            mean_bar(size(x, 1)), variance_bar(size(x, 1)), local_gradient(self%models(1)%parameter_count()), &
            kl_gradient(self%models(1)%parameter_count()))
        call self%predict_latent(x, means, variances, status)
        if (status%code /= FORTNUM_OK) return
        call logits_and_probabilities(self, means, variances, logits, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call observation_weights(size(labels), sample_weight, weights, status)
        if (status%code /= FORTNUM_OK) return
        likelihood = 0.0_dp
        mean_bar = 0.0_dp
        variance_bar = 0.0_dp
        if (self%likelihood == GP_VARIATIONAL_PROBIT) then
            c = 1.0_dp
        else
            c = PI/8.0_dp
        end if
        do i = 1, size(labels)
            class_index = find_class(self%class_label, labels(i))
            likelihood = likelihood + weights(i)*log(max(probabilities(i, class_index), tiny(1.0_dp)))
        end do
        first = 1
        do j = 1, self%n_classes
            local_count = self%models(j)%parameter_count()
            last = first + local_count - 1
            mean_bar = 0.0_dp
            variance_bar = 0.0_dp
            do i = 1, size(labels)
                class_index = find_class(self%class_label, labels(i))
                scale_value = sqrt(1.0_dp + c*variances(i, j))
                term = weights(i)*multiplier*(merge(1.0_dp, 0.0_dp, j == class_index) - probabilities(i, j))
                mean_bar(i) = term/scale_value
                variance_bar(i) = term*(-0.5_dp*c*means(i, j)/(scale_value**3))
            end do
            call self%models(j)%predict_latent_parameter_vjp(x, mean_bar, variance_bar, &
                local_gradient, status)
            if (status%code /= FORTNUM_OK) return
            call self%models(j)%kl_gradient(kl_gradient, status)
            if (status%code /= FORTNUM_OK) return
            gradient(first:last) = local_gradient - kl_gradient
            first = last + 1
        end do
        divergence = 0.0_dp
        do j = 1, self%n_classes
            call self%models(j)%kl(term, status)
            if (status%code /= FORTNUM_OK) return
            divergence = divergence + term
        end do
        value = multiplier*likelihood - divergence
        if (any(.not. ieee_is_finite(gradient)) .or. .not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: ELBO gradient is not finite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvcc_elbo_gradient

    subroutine gvcc_elbo_jvp(self, x, labels, direction, value, tangent, status, scale, &
            sample_weight)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value, tangent
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: scale
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable :: gradient(:)

        tangent = 0.0_dp
        if (size(direction) /= self%parameter_count() .or. &
                any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: JVP direction is invalid")
            value = 0.0_dp
            return
        end if
        allocate(gradient(size(direction)))
        if (present(scale) .and. present(sample_weight)) then
            call self%elbo_gradient(x, labels, value, gradient, status, scale=scale, &
                sample_weight=sample_weight)
        else if (present(scale)) then
            call self%elbo_gradient(x, labels, value, gradient, status, scale=scale)
        else if (present(sample_weight)) then
            call self%elbo_gradient(x, labels, value, gradient, status, sample_weight=sample_weight)
        else
            call self%elbo_gradient(x, labels, value, gradient, status)
        end if
        if (status%code /= FORTNUM_OK) return
        tangent = dot_product(gradient, direction)
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvcc_elbo_jvp

    subroutine gvcc_predict_latent(self, x, means, variances, status)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: means(:, :), variances(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: local_mean(:), local_variance(:)
        integer :: i

        if (.not. prediction_valid(self, x, means, variances, status)) return
        allocate(local_mean(size(x, 1)), local_variance(size(x, 1)))
        do i = 1, self%n_classes
            call self%models(i)%predict_latent(x, local_mean, local_variance, status)
            if (status%code /= FORTNUM_OK) return
            means(:, i) = local_mean
            variances(:, i) = local_variance
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvcc_predict_latent

    subroutine gvcc_predict_proba(self, x, probabilities, status)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: means(:, :), variances(:, :), logits(:, :)

        if (.not. probability_valid(self, x, probabilities, status)) return
        allocate(means(size(x, 1), self%n_classes), variances(size(x, 1), self%n_classes), &
            logits(size(x, 1), self%n_classes))
        call self%predict_latent(x, means, variances, status)
        if (status%code /= FORTNUM_OK) return
        call logits_and_probabilities(self, means, variances, logits, probabilities, status)
    end subroutine gvcc_predict_proba

    subroutine gvcc_predict(self, x, labels, status)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i, j, best

        labels = 0
        if (.not. prediction_label_valid(self, x, labels, status)) return
        allocate(probabilities(size(x, 1), self%n_classes))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(x, 1)
            best = 1
            do j = 2, self%n_classes
                if (probabilities(i, j) > probabilities(i, best)) best = j
            end do
            labels(i) = self%class_label(best)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvcc_predict

    subroutine gvcc_predict_proba_parameter_jvp(self, x, direction, probabilities, &
            probabilities_dot, status)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), direction(:)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: means(:, :), variances(:, :), logits(:, :), logits_dot(:, :)
        real(dp), allocatable :: mean_dot(:), variance_dot(:)
        integer :: i, j, first, last, local_count
        real(dp) :: c, scale_value

        if (.not. probability_valid(self, x, probabilities, status)) return
        if (size(direction) /= self%parameter_count() .or. &
                any(.not. ieee_is_finite(direction)) .or. any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP probability JVP: shape is invalid")
            return
        end if
        allocate(means(size(x, 1), self%n_classes), variances(size(x, 1), self%n_classes), &
            logits(size(x, 1), self%n_classes), logits_dot(size(x, 1), self%n_classes), &
            mean_dot(size(x, 1)), variance_dot(size(x, 1)))
        if (self%likelihood == GP_VARIATIONAL_PROBIT) then
            c = 1.0_dp
        else
            c = PI/8.0_dp
        end if
        first = 1
        do j = 1, self%n_classes
            local_count = self%models(j)%parameter_count()
            last = first + local_count - 1
            call self%models(j)%predict_latent_parameter_jvp(x, direction(first:last), &
                means(:, j), mean_dot, variances(:, j), variance_dot, status)
            if (status%code /= FORTNUM_OK) return
            do i = 1, size(x, 1)
                scale_value = sqrt(1.0_dp + c*variances(i, j))
                logits(i, j) = means(i, j)/scale_value
                logits_dot(i, j) = mean_dot(i)/scale_value - &
                    0.5_dp*c*means(i, j)*variance_dot(i)/(scale_value**3)
            end do
            first = last + 1
        end do
        call softmax_jvp(logits, logits_dot, probabilities, probabilities_dot, status)
    end subroutine gvcc_predict_proba_parameter_jvp

    subroutine gvcc_predict_proba_parameter_vjp(self, x, probabilities_bar, parameter_bar, status)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: means(:, :), variances(:, :), logits(:, :), probabilities(:, :)
        real(dp), allocatable :: logits_bar(:, :), mean_bar(:), variance_bar(:), local_bar(:)
        integer :: i, j, first, last, local_count
        real(dp) :: c, scale_value

        parameter_bar = 0.0_dp
        if (.not. probability_cotangent_valid(self, x, probabilities_bar, parameter_bar, status)) return
        allocate(means(size(x, 1), self%n_classes), variances(size(x, 1), self%n_classes), &
            logits(size(x, 1), self%n_classes), probabilities(size(x, 1), self%n_classes), &
            logits_bar(size(x, 1), self%n_classes), mean_bar(size(x, 1)), variance_bar(size(x, 1)), &
            local_bar(self%models(1)%parameter_count()))
        call self%predict_latent(x, means, variances, status)
        if (status%code /= FORTNUM_OK) return
        call logits_and_probabilities(self, means, variances, logits, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call softmax_vjp(probabilities, probabilities_bar, logits_bar)
        if (self%likelihood == GP_VARIATIONAL_PROBIT) then
            c = 1.0_dp
        else
            c = PI/8.0_dp
        end if
        first = 1
        do j = 1, self%n_classes
            local_count = self%models(j)%parameter_count()
            last = first + local_count - 1
            do i = 1, size(x, 1)
                scale_value = sqrt(1.0_dp + c*variances(i, j))
                mean_bar(i) = logits_bar(i, j)/scale_value
                variance_bar(i) = logits_bar(i, j)*(-0.5_dp*c*means(i, j)/(scale_value**3))
            end do
            call self%models(j)%predict_latent_parameter_vjp(x, mean_bar, variance_bar, &
                local_bar, status)
            if (status%code /= FORTNUM_OK) return
            parameter_bar(first:last) = local_bar
            first = last + 1
        end do
        if (any(.not. ieee_is_finite(parameter_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP probability VJP: nonfinite result")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvcc_predict_proba_parameter_vjp

    subroutine gvcc_predict_proba_input_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: means(:, :), variances(:, :), logits(:, :), logits_dot(:, :)
        real(dp), allocatable :: mean_dot(:), variance_dot(:)
        integer :: i, j
        real(dp) :: c, scale_value

        if (.not. probability_valid(self, x, probabilities, status)) return
        if (any(shape(x_dot) /= shape(x)) .or. any(shape(probabilities_dot) /= shape(probabilities)) .or. &
                any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP input JVP: shape is invalid")
            return
        end if
        allocate(means(size(x, 1), self%n_classes), variances(size(x, 1), self%n_classes), &
            logits(size(x, 1), self%n_classes), logits_dot(size(x, 1), self%n_classes), &
            mean_dot(size(x, 1)), variance_dot(size(x, 1)))
        if (self%likelihood == GP_VARIATIONAL_PROBIT) then
            c = 1.0_dp
        else
            c = PI/8.0_dp
        end if
        do j = 1, self%n_classes
            call self%models(j)%predict_latent_input_jvp(x, x_dot, means(:, j), mean_dot, &
                variances(:, j), variance_dot, status)
            if (status%code /= FORTNUM_OK) return
            do i = 1, size(x, 1)
                scale_value = sqrt(1.0_dp + c*variances(i, j))
                logits(i, j) = means(i, j)/scale_value
                logits_dot(i, j) = mean_dot(i)/scale_value - &
                    0.5_dp*c*means(i, j)*variance_dot(i)/(scale_value**3)
            end do
        end do
        call softmax_jvp(logits, logits_dot, probabilities, probabilities_dot, status)
    end subroutine gvcc_predict_proba_input_jvp

    subroutine gvcc_predict_proba_input_vjp(self, x, probabilities_bar, x_bar, status)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: means(:, :), variances(:, :), logits(:, :), probabilities(:, :)
        real(dp), allocatable :: logits_bar(:, :), mean_bar(:), variance_bar(:), local_x_bar(:, :)
        integer :: i, j
        real(dp) :: c, scale_value

        x_bar = 0.0_dp
        if (.not. probability_input_cotangent_valid(self, x, probabilities_bar, x_bar, status)) return
        allocate(means(size(x, 1), self%n_classes), variances(size(x, 1), self%n_classes), &
            logits(size(x, 1), self%n_classes), probabilities(size(x, 1), self%n_classes), &
            logits_bar(size(x, 1), self%n_classes), mean_bar(size(x, 1)), variance_bar(size(x, 1)), &
            local_x_bar(size(x, 1), size(x, 2)))
        call self%predict_latent(x, means, variances, status)
        if (status%code /= FORTNUM_OK) return
        call logits_and_probabilities(self, means, variances, logits, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        call softmax_vjp(probabilities, probabilities_bar, logits_bar)
        if (self%likelihood == GP_VARIATIONAL_PROBIT) then
            c = 1.0_dp
        else
            c = PI/8.0_dp
        end if
        do j = 1, self%n_classes
            do i = 1, size(x, 1)
                scale_value = sqrt(1.0_dp + c*variances(i, j))
                mean_bar(i) = logits_bar(i, j)/scale_value
                variance_bar(i) = logits_bar(i, j)*(-0.5_dp*c*means(i, j)/(scale_value**3))
            end do
            call self%models(j)%predict_latent_input_vjp(x, mean_bar, variance_bar, &
                local_x_bar, status)
            if (status%code /= FORTNUM_OK) return
            x_bar = x_bar + local_x_bar
        end do
        if (any(.not. ieee_is_finite(x_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP input VJP: nonfinite result")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine gvcc_predict_proba_input_vjp

    subroutine gvcc_predict_proba_device(self, device, x, probabilities, status)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        probabilities = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "categorical variational GP device: resident CUDA graph is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP device: device kind is invalid")
        end select
    end subroutine gvcc_predict_proba_device

    subroutine gvcc_predict_proba_parameter_vjp_device(self, device, x, probabilities_bar, &
            parameter_bar, status)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        parameter_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP parameter VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_parameter_vjp(x, probabilities_bar, parameter_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "categorical variational GP parameter VJP device: resident CUDA graph is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP parameter VJP device: device kind is invalid")
        end select
    end subroutine gvcc_predict_proba_parameter_vjp_device

    subroutine gvcc_predict_proba_input_vjp_device(self, device, x, probabilities_bar, x_bar, status)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        x_bar = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP input VJP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba_input_vjp(x, probabilities_bar, x_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "categorical variational GP input VJP device: resident CUDA graph is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP input VJP device: device kind is invalid")
        end select
    end subroutine gvcc_predict_proba_input_vjp_device

    subroutine gvcc_elbo_device(self, device, x, labels, value, status, scale, sample_weight)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: scale
        real(dp), intent(in), optional :: sample_weight(:)

        value = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            if (present(scale) .and. present(sample_weight)) then
                call self%elbo(x, labels, value, status, scale=scale, sample_weight=sample_weight)
            else if (present(scale)) then
                call self%elbo(x, labels, value, status, scale=scale)
            else if (present(sample_weight)) then
                call self%elbo(x, labels, value, status, sample_weight=sample_weight)
            else
                call self%elbo(x, labels, value, status)
            end if
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "categorical variational GP device: resident CUDA inducing graph is not linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP device: device kind is invalid")
        end select
    end subroutine gvcc_elbo_device

    integer function gvcc_class_count(self) result(count)
        class(gp_variational_categorical_classification_t), intent(in) :: self

        count = self%n_classes
    end function gvcc_class_count

    integer function gvcc_feature_count(self) result(count)
        class(gp_variational_categorical_classification_t), intent(in) :: self

        count = self%n_features
    end function gvcc_feature_count

    function gvcc_classes(self) result(classes)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        integer, allocatable :: classes(:)

        if (allocated(self%class_label)) then
            allocate(classes(size(self%class_label)), source=self%class_label)
        else
            allocate(classes(0))
        end if
    end function gvcc_classes

    logical function gvcc_initialized(self) result(initialized)
        class(gp_variational_categorical_classification_t), intent(in) :: self

        initialized = self%is_initialized .and. allocated(self%models) .and. &
            allocated(self%class_label) .and. self%n_classes >= 2
    end function gvcc_initialized

    logical function gvcc_device_supported(self, device_kind) result(supported)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = device_kind == FORTML_DEVICE_CPU
    end function gvcc_device_supported

    subroutine categorical_objective(context_any, parameters, value, gradient, status)
        class(*), intent(inout) :: context_any
        real(dp), intent(in) :: parameters(:)
        real(dp), intent(out) :: value, gradient(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: elbo

        value = huge(1.0_dp)
        gradient = 0.0_dp
        select type (context => context_any)
        type is (categorical_context_t)
            if (.not. associated(context%model)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "categorical variational GP objective: model is absent")
                return
            end if
            call context%model%set_parameters(parameters, status)
            if (status%code /= FORTNUM_OK) return
            if (allocated(context%sample_weight)) then
                call context%model%elbo_gradient(context%x, context%labels, elbo, gradient, &
                    status, sample_weight=context%sample_weight)
            else
                call context%model%elbo_gradient(context%x, context%labels, elbo, gradient, status)
            end if
            if (status%code /= FORTNUM_OK) return
            value = -elbo
            gradient = -gradient
            call status_set(status, FORTNUM_OK, "")
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP objective: context type is invalid")
        end select
    end subroutine categorical_objective

    subroutine copy_lbfgsb_options(options, output)
        type(gp_variational_categorical_options_t), intent(in) :: options
        type(lbfgsb_options_t), intent(out) :: output

        output%memory = options%memory
        output%max_iterations = options%max_iterations
        output%max_line_search = options%max_line_search
        output%gradient_tolerance = options%gradient_tolerance
        output%step_tolerance = options%step_tolerance
        output%objective_tolerance = options%objective_tolerance
    end subroutine copy_lbfgsb_options

    logical function valid_options(options) result(valid)
        type(gp_variational_categorical_options_t), intent(in) :: options

        valid = options%memory >= 1 .and. options%max_iterations >= 1 .and. &
            options%max_line_search >= 1 .and. options%n_mc_samples >= 1 .and. &
            options%likelihood >= GP_VARIATIONAL_LOGISTIC .and. &
            options%likelihood <= GP_VARIATIONAL_PROBIT .and. &
            ieee_is_finite(options%gradient_tolerance) .and. &
            ieee_is_finite(options%step_tolerance) .and. &
            ieee_is_finite(options%objective_tolerance) .and. ieee_is_finite(options%jitter) .and. &
            ieee_is_finite(options%lower_bound) .and. ieee_is_finite(options%upper_bound) .and. &
            options%gradient_tolerance >= 0.0_dp .and. options%step_tolerance >= 0.0_dp .and. &
            options%objective_tolerance >= 0.0_dp .and. options%jitter > 0.0_dp .and. &
            options%lower_bound <= options%upper_bound
    end function valid_options

    logical function valid_training_data(x, labels, sample_weight) result(valid)
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp) :: mass

        valid = size(x, 1) >= 1 .and. size(x, 2) >= 1 .and. size(labels) == size(x, 1) .and. &
            all(ieee_is_finite(x))
        if (.not. valid) return
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels) .or. any(.not. ieee_is_finite(sample_weight)) .or. &
                    any(sample_weight < 0.0_dp)) then
                valid = .false.
                return
            end if
            mass = sum(sample_weight)
            valid = ieee_is_finite(mass) .and. mass > 0.0_dp
        end if
    end function valid_training_data

    subroutine observation_weights(n, sample_weight, weights, status)
        integer, intent(in) :: n
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), allocatable, intent(out) :: weights(:)
        type(fortnum_status_t), intent(out) :: status

        allocate(weights(n))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n .or. any(.not. ieee_is_finite(sample_weight)) .or. &
                    any(sample_weight < 0.0_dp) .or. sum(sample_weight) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "categorical variational GP: sample weights are invalid")
                return
            end if
            weights = sample_weight
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine observation_weights

    logical function valid_data(self, x, labels, status, sample_weight) result(valid)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: sample_weight(:)
        integer :: i

        valid = self%initialized()
        if (.not. valid) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: model is not initialized")
            return
        end if
        valid = size(x, 1) >= 1 .and. size(x, 2) == self%n_features .and. &
            size(labels) == size(x, 1) .and. all(ieee_is_finite(x))
        if (.not. valid) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP: data shape is invalid")
            return
        end if
        do i = 1, size(labels)
            if (find_class(self%class_label, labels(i)) < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "categorical variational GP: unknown label")
                valid = .false.
                return
            end if
        end do
        if (present(sample_weight)) then
            if (size(sample_weight) /= size(labels)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "categorical variational GP: sample weights are invalid")
                valid = .false.
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "categorical variational GP: sample weights are invalid")
                valid = .false.
                return
            end if
            if (any(sample_weight < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "categorical variational GP: sample weights are invalid")
                valid = .false.
                return
            end if
            if (sum(sample_weight) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "categorical variational GP: sample weights are invalid")
                valid = .false.
                return
            end if
        end if
        call status_set(status, FORTNUM_OK, "")
    end function valid_data

    logical function prediction_valid(self, x, means, variances, status) result(valid)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), means(:, :), variances(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = self%initialized()
        if (.not. valid) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP prediction: model is not initialized")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
                any(shape(means) /= [size(x, 1), self%n_classes]) .or. &
                any(shape(variances) /= shape(means)) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP prediction: shape is invalid")
            valid = .false.
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end function prediction_valid

    logical function probability_valid(self, x, probabilities, status) result(valid)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = self%initialized()
        if (.not. valid) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP probability prediction: model is not initialized")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
                any(shape(probabilities) /= [size(x, 1), self%n_classes]) .or. &
                any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP probability prediction: shape is invalid")
            valid = .false.
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end function probability_valid

    logical function prediction_label_valid(self, x, labels, status) result(valid)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        valid = self%initialized()
        if (.not. valid) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP prediction: model is not initialized")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
                size(labels) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP prediction: label shape is invalid")
            valid = .false.
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end function prediction_label_valid

    logical function probability_cotangent_valid(self, x, probabilities_bar, parameter_bar, status) result(valid)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :), parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        valid = probability_valid(self, x, probabilities_bar, status)
        if (.not. valid) return
        if (size(parameter_bar) /= self%parameter_count() .or. &
                any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP probability VJP: shape is invalid")
            valid = .false.
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end function probability_cotangent_valid

    logical function probability_input_cotangent_valid(self, x, probabilities_bar, x_bar, status) result(valid)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :), x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        valid = probability_valid(self, x, probabilities_bar, status)
        if (.not. valid) return
        if (any(shape(x_bar) /= shape(x)) .or. any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP input VJP: shape is invalid")
            valid = .false.
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end function probability_input_cotangent_valid

    subroutine logits_and_probabilities(self, means, variances, logits, probabilities, status)
        class(gp_variational_categorical_classification_t), intent(in) :: self
        real(dp), intent(in) :: means(:, :), variances(:, :)
        real(dp), intent(out) :: logits(:, :), probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: c, scale_value
        integer :: i, j

        if (self%likelihood == GP_VARIATIONAL_PROBIT) then
            c = 1.0_dp
        else
            c = PI/8.0_dp
        end if
        do i = 1, size(means, 1)
            do j = 1, size(means, 2)
                if (variances(i, j) <= 0.0_dp .or. &
                        .not. ieee_is_finite(variances(i, j))) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "categorical variational GP: latent variance is invalid")
                    return
                end if
                scale_value = sqrt(1.0_dp + c*variances(i, j))
                logits(i, j) = means(i, j)/scale_value
            end do
        end do
        call stable_softmax(logits, probabilities, status)
    end subroutine logits_and_probabilities

    subroutine stable_softmax(logits, probabilities, status)
        real(dp), intent(in) :: logits(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: shift, total
        integer :: i, j

        if (any(shape(probabilities) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP softmax: shape is invalid")
            return
        end if
        do i = 1, size(logits, 1)
            shift = maxval(logits(i, :))
            probabilities(i, :) = exp(logits(i, :) - shift)
            total = sum(probabilities(i, :))
            if (.not. ieee_is_finite(total) .or. total <= tiny(1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "categorical variational GP softmax: normalization is invalid")
                return
            end if
            probabilities(i, :) = probabilities(i, :)/total
            do j = 1, size(logits, 2)
                if (.not. ieee_is_finite(probabilities(i, j))) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "categorical variational GP softmax: probability is not finite")
                    return
                end if
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine stable_softmax

    subroutine softmax_jvp(logits, logits_dot, probabilities, probabilities_dot, status)
        real(dp), intent(in) :: logits(:, :), logits_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: mean_tangent
        integer :: i

        if (any(shape(logits_dot) /= shape(logits)) .or. &
                any(shape(probabilities) /= shape(logits)) .or. &
                any(shape(probabilities_dot) /= shape(logits))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "categorical variational GP softmax JVP: shape is invalid")
            return
        end if
        call stable_softmax(logits, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(logits, 1)
            mean_tangent = dot_product(probabilities(i, :), logits_dot(i, :))
            probabilities_dot(i, :) = probabilities(i, :)*(logits_dot(i, :) - mean_tangent)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine softmax_jvp

    subroutine softmax_vjp(probabilities, probabilities_bar, logits_bar)
        real(dp), intent(in) :: probabilities(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: logits_bar(:, :)
        real(dp) :: mean_bar
        integer :: i

        do i = 1, size(probabilities, 1)
            mean_bar = dot_product(probabilities(i, :), probabilities_bar(i, :))
            logits_bar(i, :) = probabilities(i, :)*(probabilities_bar(i, :) - mean_bar)
        end do
    end subroutine softmax_vjp

    integer function find_class(classes, label) result(index)
        integer, intent(in) :: classes(:), label
        integer :: i

        index = 0
        do i = 1, size(classes)
            if (classes(i) == label) then
                index = i
                return
            end if
        end do
    end function find_class

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
        allocate(output(n), source=work(:n))
    end subroutine sorted_unique

    function unique_labels_of(labels) result(classes)
        integer, intent(in) :: labels(:)
        integer, allocatable :: classes(:)

        call sorted_unique(labels, classes)
    end function unique_labels_of

end module fortml_gp_variational_categorical_classification
