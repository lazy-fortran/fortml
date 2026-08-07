module fortml_one_class_svm
    !! One-class support-vector machine with a dense RBF kernel.
    !!
    !! The estimator solves the standard nu-SVM dual
    !!
    !!   min_a 0.5*a^T*K*a,  0 <= a_i <= 1/(nu*n),  sum(a) = 1,
    !!
    !! using a deterministic projected-gradient iteration on the capped
    !! simplex.  The fitted state is an explicit finite training matrix,
    !! kernel weights, and offset.  Fit-time active-set derivatives are not
    !! exposed; fixed-state RBF decision JVP/VJP products are exact.  CUDA is
    !! an explicit refusal until a resident kernel reduction is linked.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    real(dp), parameter :: ONE_CLASS_SUPPORT_TOLERANCE = 1.0e-10_dp

    type, public :: one_class_svm_t
        private
        real(dp), allocatable :: x_train(:, :)
        real(dp), allocatable :: alpha(:)
        real(dp) :: gamma_value = 1.0_dp
        real(dp) :: nu_value = 0.5_dp
        real(dp) :: rho_value = 0.0_dp
        integer :: n_features = 0
        integer :: n_samples = 0
        integer :: iterations_value = 0
        logical :: is_fitted = .false.
    contains
        procedure, public :: fit => one_class_svm_fit
        procedure, public :: decision_function => one_class_svm_decision_function
        procedure, public :: predict => one_class_svm_predict
        procedure, public :: decision_function_jvp => one_class_svm_decision_jvp
        procedure, public :: decision_function_vjp => one_class_svm_decision_vjp
        procedure, public :: jvp => one_class_svm_decision_jvp
        procedure, public :: vjp => one_class_svm_decision_vjp
        procedure, public :: predict_device => one_class_svm_predict_device
        procedure, public :: decision_function_device => &
            one_class_svm_decision_device
        procedure, public :: device_supported => one_class_svm_device_supported
        procedure, public :: support_weights => one_class_svm_support_weights
        procedure, public :: offset => one_class_svm_offset
        procedure, public :: gamma => one_class_svm_gamma
        procedure, public :: nu => one_class_svm_nu
        procedure, public :: support_vector_count => one_class_svm_support_count
        procedure, public :: feature_count => one_class_svm_feature_count
        procedure, public :: sample_count => one_class_svm_sample_count
        procedure, public :: iterations => one_class_svm_iterations
        procedure, public :: fitted => one_class_svm_fitted
    end type one_class_svm_t

    public :: one_class_svm_fit
    public :: one_class_svm_decision_function
    public :: one_class_svm_predict

contains

    subroutine one_class_svm_fit(self, x, status, nu, gamma, max_iterations, tolerance)
        class(one_class_svm_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: nu, gamma, tolerance
        integer, intent(in), optional :: max_iterations
        real(dp), allocatable :: kernel(:, :), alpha_new(:), alpha_old(:), gradient(:)
        real(dp) :: requested_nu, requested_gamma, requested_tolerance
        real(dp) :: cap, step, row_sum, change, rho_lower, rho_upper
        integer :: requested_iterations, i, j, iteration, n_samples
        logical :: converged

        self%is_fitted = .false.
        n_samples = size(x, 1)
        if (n_samples < 1 .or. size(x, 2) < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM fit: at least one finite sample and feature are required")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM fit: training inputs must be finite")
            return
        end if

        requested_nu = 0.5_dp
        if (present(nu)) requested_nu = nu
        if (.not. ieee_is_finite(requested_nu) .or. requested_nu <= 0.0_dp .or. &
            requested_nu > 1.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM fit: nu must lie in (0,1]")
            return
        end if

        requested_gamma = 1.0_dp/real(size(x, 2), dp)
        if (present(gamma)) requested_gamma = gamma
        if (.not. ieee_is_finite(requested_gamma) .or. requested_gamma <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM fit: gamma must be finite and positive")
            return
        end if

        requested_iterations = 5000
        if (present(max_iterations)) requested_iterations = max_iterations
        requested_tolerance = 1.0e-8_dp
        if (present(tolerance)) requested_tolerance = tolerance
        if (requested_iterations < 1 .or. &
            .not. ieee_is_finite(requested_tolerance) .or. &
            requested_tolerance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM fit: iteration limit and tolerance are invalid")
            return
        end if

        allocate(kernel(n_samples, n_samples), alpha_new(n_samples), &
            alpha_old(n_samples), gradient(n_samples))
        call make_rbf_kernel(x, requested_gamma, kernel, status)
        if (status%code /= FORTNUM_OK) return

        cap = 1.0_dp/(requested_nu*real(n_samples, dp))
        alpha_new = 1.0_dp/real(n_samples, dp)
        row_sum = 0.0_dp
        do i = 1, n_samples
            row_sum = max(row_sum, sum(kernel(i, :)))
        end do
        if (.not. ieee_is_finite(row_sum) .or. row_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM fit: kernel Lipschitz bound is invalid")
            return
        end if
        step = 0.99_dp/row_sum
        converged = .false.
        do iteration = 1, requested_iterations
            alpha_old = alpha_new
            gradient = matmul(kernel, alpha_old)
            call project_capped_simplex(alpha_old - step*gradient, cap, alpha_new)
            change = maxval(abs(alpha_new - alpha_old))
            if (change <= requested_tolerance) then
                converged = .true.
                exit
            end if
        end do
        if (.not. converged) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "one-class SVM fit: projected dual iteration did not converge")
            return
        end if

        allocate(self%x_train(n_samples, size(x, 2)))
        self%x_train = x
        allocate(self%alpha(n_samples))
        self%alpha = alpha_new
        self%gamma_value = requested_gamma
        self%nu_value = requested_nu
        self%n_features = size(x, 2)
        self%n_samples = n_samples
        self%iterations_value = iteration
        call compute_training_scores(self, self%alpha, rho_lower, rho_upper, row_sum)
        self%rho_value = choose_offset(rho_lower, rho_upper, row_sum)
        self%is_fitted = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine one_class_svm_fit

    subroutine one_class_svm_decision_function(self, x, scores, status)
        class(one_class_svm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, k
        real(dp) :: d2, delta, kernel_value

        scores = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM decision: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            size(scores) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM decision: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM decision: inputs must be finite")
            return
        end if
        do i = 1, size(x, 1)
            scores(i) = -self%rho_value
            do j = 1, self%n_samples
                d2 = 0.0_dp
                do k = 1, self%n_features
                    delta = x(i, k) - self%x_train(j, k)
                    d2 = d2 + delta*delta
                end do
                if (.not. ieee_is_finite(d2)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "one-class SVM decision: distance overflow")
                    return
                end if
                kernel_value = exp(-self%gamma_value*d2)
                scores(i) = scores(i) + self%alpha(j)*kernel_value
            end do
        end do
        if (any(.not. ieee_is_finite(scores))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM decision: score overflow")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine one_class_svm_decision_function

    subroutine one_class_svm_predict(self, x, labels, status)
        class(one_class_svm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scores(:)
        integer :: i

        labels = 0
        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM predict: output shape is invalid")
            return
        end if
        allocate(scores(size(x, 1)))
        call self%decision_function(x, scores, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            if (scores(i) >= 0.0_dp) then
                labels(i) = 1
            else
                labels(i) = -1
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine one_class_svm_predict

    subroutine one_class_svm_decision_device(self, device, x, scores, status)
        class(one_class_svm_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: scores(:)
        type(fortnum_status_t), intent(out) :: status

        scores = 0.0_dp
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM device decision: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%decision_function(x, scores, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "one-class SVM device decision: no resident RBF kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM device decision: device kind is invalid")
        end select
    end subroutine one_class_svm_decision_device

    subroutine one_class_svm_predict_device(self, device, x, labels, status)
        class(one_class_svm_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        labels = 0
        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "one-class SVM device prediction: no resident RBF kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM device prediction: device kind is invalid")
        end select
    end subroutine one_class_svm_predict_device

    subroutine one_class_svm_decision_jvp(self, x, x_dot, scores, scores_dot, status)
        class(one_class_svm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: scores(:), scores_dot(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, k
        real(dp) :: d2, delta, direction, kernel_value

        scores = 0.0_dp
        scores_dot = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM decision JVP: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            any(shape(x_dot) /= shape(x)) .or. size(scores) /= size(x, 1) .or. &
            size(scores_dot) /= size(scores)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM decision JVP: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM decision JVP: inputs must be finite")
            return
        end if
        do i = 1, size(x, 1)
            scores(i) = -self%rho_value
            do j = 1, self%n_samples
                d2 = 0.0_dp
                direction = 0.0_dp
                do k = 1, self%n_features
                    delta = x(i, k) - self%x_train(j, k)
                    d2 = d2 + delta*delta
                    direction = direction + delta*x_dot(i, k)
                end do
                if (.not. ieee_is_finite(d2) .or. &
                    .not. ieee_is_finite(direction)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "one-class SVM decision JVP: distance overflow")
                    return
                end if
                kernel_value = exp(-self%gamma_value*d2)
                scores(i) = scores(i) + self%alpha(j)*kernel_value
                scores_dot(i) = scores_dot(i) - 2.0_dp*self%gamma_value* &
                    self%alpha(j)*kernel_value*direction
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine one_class_svm_decision_jvp

    subroutine one_class_svm_decision_vjp(self, x, scores_bar, x_bar, status)
        class(one_class_svm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), scores_bar(:)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, k
        real(dp) :: d2, delta, kernel_value, coefficient

        x_bar = 0.0_dp
        if (.not. self%is_fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM decision VJP: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_features .or. &
            size(scores_bar) /= size(x, 1) .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM decision VJP: input or output shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(scores_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM decision VJP: inputs and cotangents must be finite")
            return
        end if
        do i = 1, size(x, 1)
            do j = 1, self%n_samples
                d2 = 0.0_dp
                do k = 1, self%n_features
                    delta = x(i, k) - self%x_train(j, k)
                    d2 = d2 + delta*delta
                end do
                if (.not. ieee_is_finite(d2)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "one-class SVM decision VJP: distance overflow")
                    return
                end if
                kernel_value = exp(-self%gamma_value*d2)
                coefficient = -2.0_dp*self%gamma_value*self%alpha(j)* &
                    scores_bar(i)*kernel_value
                do k = 1, self%n_features
                    x_bar(i, k) = x_bar(i, k) + coefficient* &
                        (x(i, k) - self%x_train(j, k))
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine one_class_svm_decision_vjp

    logical function one_class_svm_device_supported(self, device_kind) result(supported)
        class(one_class_svm_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = .false.
        if (device_kind == FORTML_DEVICE_CPU) supported = self%is_fitted
    end function one_class_svm_device_supported

    function one_class_svm_support_weights(self) result(values)
        class(one_class_svm_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%alpha)) then
            values = self%alpha
        else
            allocate(values(0))
        end if
    end function one_class_svm_support_weights

    real(dp) function one_class_svm_offset(self) result(value)
        class(one_class_svm_t), intent(in) :: self
        value = self%rho_value
    end function one_class_svm_offset

    real(dp) function one_class_svm_gamma(self) result(value)
        class(one_class_svm_t), intent(in) :: self
        value = self%gamma_value
    end function one_class_svm_gamma

    real(dp) function one_class_svm_nu(self) result(value)
        class(one_class_svm_t), intent(in) :: self
        value = self%nu_value
    end function one_class_svm_nu

    integer function one_class_svm_support_count(self) result(value)
        class(one_class_svm_t), intent(in) :: self
        value = 0
        if (allocated(self%alpha)) then
            value = count(self%alpha > ONE_CLASS_SUPPORT_TOLERANCE)
        end if
    end function one_class_svm_support_count

    integer function one_class_svm_feature_count(self) result(value)
        class(one_class_svm_t), intent(in) :: self
        value = self%n_features
    end function one_class_svm_feature_count

    integer function one_class_svm_sample_count(self) result(value)
        class(one_class_svm_t), intent(in) :: self
        value = self%n_samples
    end function one_class_svm_sample_count

    integer function one_class_svm_iterations(self) result(value)
        class(one_class_svm_t), intent(in) :: self
        value = self%iterations_value
    end function one_class_svm_iterations

    logical function one_class_svm_fitted(self) result(value)
        class(one_class_svm_t), intent(in) :: self
        value = self%is_fitted .and. allocated(self%alpha) .and. &
            allocated(self%x_train)
    end function one_class_svm_fitted

    subroutine make_rbf_kernel(x, gamma, kernel, status)
        real(dp), intent(in) :: x(:, :), gamma
        real(dp), intent(out) :: kernel(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, k
        real(dp) :: d2, delta

        kernel = 0.0_dp
        if (any(shape(kernel) /= [size(x, 1), size(x, 1)])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM kernel: output shape is invalid")
            return
        end if
        do i = 1, size(x, 1)
            do j = 1, size(x, 1)
                d2 = 0.0_dp
                do k = 1, size(x, 2)
                    delta = x(i, k) - x(j, k)
                    d2 = d2 + delta*delta
                end do
                if (.not. ieee_is_finite(d2)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "one-class SVM kernel: distance overflow")
                    return
                end if
                kernel(i, j) = exp(-gamma*d2)
            end do
        end do
        if (any(.not. ieee_is_finite(kernel))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "one-class SVM kernel: nonfinite RBF values")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine make_rbf_kernel

    subroutine project_capped_simplex(values, cap, projected)
        real(dp), intent(in) :: values(:), cap
        real(dp), intent(out) :: projected(:)
        real(dp) :: lower, upper, midpoint, mass
        integer :: iteration

        lower = minval(values) - cap
        upper = maxval(values)
        do iteration = 1, 100
            midpoint = 0.5_dp*(lower + upper)
            projected = min(max(values - midpoint, 0.0_dp), cap)
            mass = sum(projected)
            if (mass > 1.0_dp) then
                lower = midpoint
            else
                upper = midpoint
            end if
        end do
        midpoint = 0.5_dp*(lower + upper)
        projected = min(max(values - midpoint, 0.0_dp), cap)
    end subroutine project_capped_simplex

    subroutine compute_training_scores(self, alpha, lower, upper, mean_score)
        class(one_class_svm_t), intent(in) :: self
        real(dp), intent(in) :: alpha(:)
        real(dp), intent(out) :: lower, upper, mean_score
        real(dp), allocatable :: scores(:)
        real(dp), allocatable :: kernel(:, :)
        integer :: i, free_count
        real(dp) :: cap
        type(fortnum_status_t) :: status

        allocate(scores(self%n_samples))
        allocate(kernel(self%n_samples, self%n_samples))
        call make_rbf_kernel(self%x_train, self%gamma_value, kernel, status)
        if (status%code /= FORTNUM_OK) then
            scores = 0.0_dp
        else
            scores = matmul(kernel, alpha)
        end if
        mean_score = sum(scores)/real(self%n_samples, dp)
        cap = 1.0_dp/(self%nu_value*real(self%n_samples, dp))
        lower = -huge(1.0_dp)
        upper = huge(1.0_dp)
        free_count = 0
        do i = 1, self%n_samples
            if (alpha(i) > ONE_CLASS_SUPPORT_TOLERANCE .and. &
                alpha(i) < cap-ONE_CLASS_SUPPORT_TOLERANCE) then
                lower = max(lower, scores(i))
                upper = min(upper, scores(i))
                free_count = free_count + 1
            else if (alpha(i) <= ONE_CLASS_SUPPORT_TOLERANCE) then
                lower = max(lower, scores(i))
            else
                upper = min(upper, scores(i))
            end if
        end do
        if (free_count == 0) then
            ! No free support vector exists.  The admissible offset interval is
            ! used when nonempty; choose its midpoint deterministically.
            if (.not. ieee_is_finite(lower)) lower = upper
            if (.not. ieee_is_finite(upper)) upper = lower
        end if
    end subroutine compute_training_scores

    real(dp) function choose_offset(lower, upper, mean_score) result(offset)
        real(dp), intent(in) :: lower, upper, mean_score

        if (.not. ieee_is_finite(lower)) then
            offset = mean_score
            return
        end if
        if (.not. ieee_is_finite(upper)) then
            offset = mean_score
            return
        end if
        if (lower <= upper) then
            offset = 0.5_dp*(lower + upper)
            return
        end if
        offset = mean_score
    end function choose_offset

end module fortml_one_class_svm
