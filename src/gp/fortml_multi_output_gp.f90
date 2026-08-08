module fortml_multi_output_gp
    !! Correlated multi-output Gaussian process through the intrinsic
    !! coregionalization model.
    !!
    !! The joint covariance of `p` outputs over `n` inputs is
    !! `B (x) K`, where `K` is the shared input kernel and `B = W W^T +
    !! diag(kappa)` is the `p x p` coregionalization matrix. `W` has one column
    !! per latent process, so a rank-one `W` couples every output through a
    !! single shared function while `kappa` keeps an independent part per
    !! output. Setting `W = 0` makes `B` diagonal and the outputs independent,
    !! which is the degenerate case the tests use to compare against separate
    !! single-output fits.
    !!
    !! Ordering is output-major: entry `(j - 1)*n + i` is output `j` at input
    !! `i`, so the input block is contiguous.
    use fortnum_kinds, only: dp
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortml_kernels, only: kernel_t
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    implicit none
    private

    real(dp), parameter :: PI = 3.141592653589793238462643_dp

    type, public :: multi_output_gp_t
        type(kernel_t) :: kernel
        real(dp), allocatable :: inputs(:, :)
        real(dp), allocatable :: coregionalization(:, :)
        real(dp), allocatable :: weights(:, :)
        real(dp), allocatable :: independent(:)
        real(dp), allocatable :: alpha(:)
        type(cholesky_factorization_t) :: factorization
        real(dp) :: noise_variance = 1.0_dp
        integer :: n_outputs = 0
        integer :: n_samples = 0
        logical :: fitted = .false.
    contains
        procedure, public :: initialize => multi_output_initialize
        procedure, public :: fit => multi_output_fit
        procedure, public :: predict => multi_output_predict
        procedure, public :: predict_batch => multi_output_predict_batch
        procedure, public :: parameter_count => multi_output_parameter_count
        procedure, public :: parameters => multi_output_parameters
        procedure, public :: predict_input_jvp => multi_output_predict_input_jvp
        procedure, public :: predict_input_vjp => multi_output_predict_input_vjp
        procedure, public :: predict_batch_input_jvp => &
            multi_output_predict_batch_input_jvp
        procedure, public :: predict_batch_input_vjp => &
            multi_output_predict_batch_input_vjp
        procedure, public :: predict_parameter_jvp => multi_output_predict_parameter_jvp
        procedure, public :: predict_parameter_vjp => multi_output_predict_parameter_vjp
        procedure, public :: predict_input_jvp_device => &
            multi_output_predict_input_jvp_device
        procedure, public :: predict_input_vjp_device => &
            multi_output_predict_input_vjp_device
        procedure, public :: predict_batch_device => multi_output_predict_batch_device
        procedure, public :: predict_batch_input_jvp_device => &
            multi_output_predict_batch_input_jvp_device
        procedure, public :: predict_batch_input_vjp_device => &
            multi_output_predict_batch_input_vjp_device
        procedure, public :: predict_parameter_jvp_device => &
            multi_output_predict_parameter_jvp_device
        procedure, public :: predict_parameter_vjp_device => &
            multi_output_predict_parameter_vjp_device
        procedure, public :: log_marginal_likelihood => multi_output_lml
        procedure, public :: joint_covariance => multi_output_joint_covariance
    end type multi_output_gp_t

contains

    subroutine multi_output_initialize(self, kernel, weights, independent, &
            noise_variance, status)
        class(multi_output_gp_t), intent(out) :: self
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: weights(:, :)
        real(dp), intent(in) :: independent(:)
        real(dp), intent(in) :: noise_variance
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j

        if (size(weights, 1) < 1 .or. size(weights, 2) < 1 .or. &
            size(independent) /= size(weights, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP: coregionalization shape is invalid")
            return
        end if
        if (any(independent < 0.0_dp) .or. noise_variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP: variances must be non-negative and noise positive")
            return
        end if
        self%kernel = kernel
        self%noise_variance = noise_variance
        self%n_outputs = size(weights, 1)
        allocate(self%weights, source=weights)
        allocate(self%independent, source=independent)
        allocate(self%coregionalization(self%n_outputs, self%n_outputs))
        do j = 1, self%n_outputs
            do i = 1, self%n_outputs
                self%coregionalization(i, j) = sum(weights(i, :)*weights(j, :))
            end do
            self%coregionalization(j, j) = self%coregionalization(j, j) + &
                independent(j)
        end do
        self%fitted = .false.
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_output_initialize

    subroutine multi_output_joint_covariance(self, inputs, matrix, status)
        !! `B (x) K` on the given inputs, without observation noise.
        class(multi_output_gp_t), intent(in) :: self
        real(dp), intent(in) :: inputs(:, :)
        real(dp), intent(out) :: matrix(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: block(:, :)
        integer :: n, p, i, j, a, b

        n = size(inputs, 1)
        p = self%n_outputs
        if (p < 1 .or. any(shape(matrix) /= [n*p, n*p])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP: joint covariance shape is invalid")
            return
        end if
        allocate(block(n, n))
        call self%kernel%matrix(inputs, inputs, block, status)
        if (status%code /= FORTNUM_OK) return
        do b = 1, p
            do a = 1, p
                do j = 1, n
                    do i = 1, n
                        matrix((a - 1)*n + i, (b - 1)*n + j) = &
                            self%coregionalization(a, b)*block(i, j)
                    end do
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_output_joint_covariance

    subroutine multi_output_fit(self, inputs, targets, status)
        !! `targets(i, j)` is output `j` at input `i`.
        class(multi_output_gp_t), intent(inout) :: self
        real(dp), intent(in) :: inputs(:, :), targets(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: joint(:, :), stacked(:)
        integer :: n, p, i, j

        self%fitted = .false.
        n = size(inputs, 1)
        p = self%n_outputs
        if (p < 1 .or. n < 1 .or. size(targets, 1) /= n .or. &
            size(targets, 2) /= p .or. size(inputs, 2) /= self%kernel%input_dim) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP: training shape is invalid")
            return
        end if

        if (allocated(self%inputs)) deallocate(self%inputs)
        allocate(self%inputs, source=inputs)
        self%n_samples = n
        allocate(joint(n*p, n*p), stacked(n*p))
        call multi_output_joint_covariance(self, inputs, joint, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, n*p
            joint(i, i) = joint(i, i) + self%noise_variance
        end do
        do j = 1, p
            do i = 1, n
                stacked((j - 1)*n + i) = targets(i, j)
            end do
        end do
        call self%factorization%factorize(joint, status)
        if (status%code /= FORTNUM_OK) return
        if (allocated(self%alpha)) deallocate(self%alpha)
        allocate(self%alpha, source=stacked)
        call self%factorization%solve(self%alpha, status)
        if (status%code /= FORTNUM_OK) return
        self%fitted = .true.
    end subroutine multi_output_fit

    subroutine multi_output_predict(self, query, mean, status)
        !! Posterior mean, `mean(i, j)` for output `j` at query input `i`.
        class(multi_output_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :)
        real(dp), intent(out) :: mean(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: cross(:, :), block(:, :), product(:)
        integer :: n, p, m, i, j, a, b

        mean = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP: predict before fit")
            return
        end if
        n = self%n_samples
        p = self%n_outputs
        m = size(query, 1)
        if (size(query, 2) /= self%kernel%input_dim .or. &
            size(mean, 1) /= m .or. size(mean, 2) /= p) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP: prediction shape is invalid")
            return
        end if

        allocate(block(m, n), cross(m*p, n*p), product(m*p))
        call self%kernel%matrix(query, self%inputs, block, status)
        if (status%code /= FORTNUM_OK) return
        do b = 1, p
            do a = 1, p
                do j = 1, n
                    do i = 1, m
                        cross((a - 1)*m + i, (b - 1)*n + j) = &
                            self%coregionalization(a, b)*block(i, j)
                    end do
                end do
            end do
        end do
        product = matmul(cross, self%alpha)
        do j = 1, p
            do i = 1, m
                mean(i, j) = product((j - 1)*m + i)
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_output_predict

    subroutine multi_output_predict_batch(self, query, mean, status)
        !! Posterior means for a batch of independent query sets.
        !!
        !! `query(batch,point,feature)` and `mean(batch,point,output)` retain
        !! the public point-major ordering inside each batch member.  The
        !! fitted state is shared, but each batch member is shape-checked.
        class(multi_output_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :, :)
        real(dp), intent(out) :: mean(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: batch

        call multi_output_check_batch_query(self, query, mean, status)
        if (status%code /= FORTNUM_OK) return
        do batch = 1, size(query, 1)
            call self%predict(query(batch, :, :), mean(batch, :, :), status)
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_output_predict_batch

    subroutine multi_output_predict_batch_input_jvp(self, query, direction, mean, &
            mean_dot, status)
        !! Query-input JVP for a batch of independent query sets.
        class(multi_output_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :, :), direction(:, :, :)
        real(dp), intent(out) :: mean(:, :, :), mean_dot(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: batch

        call multi_output_check_batch_query(self, query, mean, status)
        if (status%code /= FORTNUM_OK) return
        if (any(shape(direction) /= shape(query)) .or. &
            any(shape(mean_dot) /= shape(mean)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP batch input JVP: direction or output shape is invalid")
            return
        end if
        do batch = 1, size(query, 1)
            call self%predict_input_jvp(query(batch, :, :), direction(batch, :, :), &
                mean(batch, :, :), mean_dot(batch, :, :), status)
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_output_predict_batch_input_jvp

    subroutine multi_output_predict_batch_input_vjp(self, query, mean_bar, query_bar, &
            status)
        !! Query-input VJP for a batch of independent query sets.
        class(multi_output_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :, :), mean_bar(:, :, :)
        real(dp), intent(out) :: query_bar(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: batch

        query_bar = 0.0_dp
        call multi_output_check_batch_query(self, query, mean_bar, status)
        if (status%code /= FORTNUM_OK) return
        if (any(shape(query_bar) /= shape(query)) .or. &
            any(.not. ieee_is_finite(mean_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP batch input VJP: cotangent or output shape is invalid")
            return
        end if
        do batch = 1, size(query, 1)
            call self%predict_input_vjp(query(batch, :, :), mean_bar(batch, :, :), &
                query_bar(batch, :, :), status)
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_output_predict_batch_input_vjp

    integer function multi_output_parameter_count(self) result(count)
        !! Number of packed fitted-model coordinates.
        !!
        !! The order is ``kernel parameters, log(noise variance), weights,
        !! independent``.  The weight block is output-major (all latent
        !! weights for output one, then output two, ...), matching the public
        !! `(sample,output)` prediction convention; the final independent
        !! block has one coordinate per output.
        class(multi_output_gp_t), intent(in) :: self
        integer :: rank

        rank = 0
        if (allocated(self%weights)) rank = size(self%weights, 2)
        count = self%kernel%parameter_count() + 1 + self%n_outputs*rank + &
            self%n_outputs
    end function multi_output_parameter_count

    function multi_output_parameters(self) result(parameters)
        !! Return the packed model coordinates described by
        !! `multi_output_parameter_count`.
        class(multi_output_gp_t), intent(in) :: self
        real(dp), allocatable :: parameters(:)
        integer :: kernel_count, rank, a, ell, index

        kernel_count = self%kernel%parameter_count()
        rank = 0
        if (allocated(self%weights)) rank = size(self%weights, 2)
        allocate(parameters(self%parameter_count()))
        parameters = 0.0_dp
        if (kernel_count > 0) parameters(:kernel_count) = self%kernel%parameters()
        parameters(kernel_count + 1) = log(self%noise_variance)
        index = kernel_count + 2
        do a = 1, self%n_outputs
            do ell = 1, rank
                parameters(index) = self%weights(a, ell)
                index = index + 1
            end do
        end do
        if (self%n_outputs > 0) parameters(index:) = self%independent
    end function multi_output_parameters

    subroutine multi_output_predict_input_jvp(self, query, direction, mean, mean_dot, status)
        !! Query-input JVP of the fitted posterior mean.
        !!
        !! The fit state (`alpha` and the training factorization) is held
        !! fixed.  Only the query locations move, and the returned arrays use
        !! output-major rows internally while retaining the public
        !! `(query,output)` shape.  Kernel input derivatives are analytic;
        !! unsupported user kernels return their typed kernel status.
        class(multi_output_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :), direction(:, :)
        real(dp), intent(out) :: mean(:, :), mean_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: base(:, :), base_dot(:, :), cross_dot(:, :)
        real(dp) :: value, grad_x1(self%kernel%input_dim), grad_x2(self%kernel%input_dim)
        real(dp), allocatable :: hessian(:, :)
        integer :: n, m, p, d, i, j, a, b, out_i, out_j

        call multi_output_check_query(self, query, mean, status)
        if (status%code /= FORTNUM_OK) return
        m = size(query, 1)
        d = size(query, 2)
        if (any(shape(direction) /= shape(query)) .or. any(shape(mean_dot) /= shape(mean)) .or. &
            any(.not. ieee_is_finite(direction))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP input JVP: direction or output shape is invalid")
            return
        end if
        n = self%n_samples
        p = self%n_outputs
        allocate(base(n, m), base_dot(n, m), cross_dot(m*p, n*p))
        allocate(hessian(d, d))
        base_dot = 0.0_dp
        do j = 1, m
            do i = 1, n
                call self%kernel%input_derivatives(query(j, :), self%inputs(i, :), &
                    value, grad_x1, grad_x2, hessian, status)
                if (status%code /= FORTNUM_OK) return
                base(i, j) = value
                base_dot(i, j) = dot_product(grad_x1, direction(j, :))
            end do
        end do
        do b = 1, p
            do a = 1, p
                do j = 1, n
                    do i = 1, m
                        out_i = (a - 1)*m + i
                        out_j = (b - 1)*n + j
                        cross_dot(out_i, out_j) = self%coregionalization(a, b)*base_dot(j, i)
                    end do
                end do
            end do
        end do
        mean = 0.0_dp
        mean_dot = 0.0_dp
        do a = 1, p
            do i = 1, m
                mean(i, a) = 0.0_dp
                do b = 1, p
                    do j = 1, n
                        mean(i, a) = mean(i, a) + self%coregionalization(a, b)* &
                            base(j, i)*self%alpha((b - 1)*n + j)
                        mean_dot(i, a) = mean_dot(i, a) + self%coregionalization(a, b)* &
                            base_dot(j, i)*self%alpha((b - 1)*n + j)
                    end do
                end do
            end do
        end do
        if (any(.not. ieee_is_finite(mean_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multi-output GP input JVP: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_output_predict_input_jvp

    subroutine multi_output_predict_input_vjp(self, query, mean_bar, query_bar, status)
        !! Adjoint of `predict_input_jvp` for a fixed fitted state.
        class(multi_output_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :), mean_bar(:, :)
        real(dp), intent(out) :: query_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: base(:, :), cross_bar(:, :)
        real(dp) :: value, grad_x1(self%kernel%input_dim), grad_x2(self%kernel%input_dim)
        real(dp), allocatable :: hessian(:, :)
        integer :: n, m, p, d, i, j, a, b, out_i, out_j, k

        query_bar = 0.0_dp
        call multi_output_check_query(self, query, mean_bar, status)
        if (status%code /= FORTNUM_OK) return
        m = size(query, 1)
        d = size(query, 2)
        if (any(shape(query_bar) /= shape(query)) .or. any(.not. ieee_is_finite(mean_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP input VJP: cotangent or output shape is invalid")
            return
        end if
        n = self%n_samples
        p = self%n_outputs
        allocate(base(n, m), cross_bar(m*p, n*p), hessian(d, d))
        do j = 1, m
            do i = 1, n
                call self%kernel%input_derivatives(query(j, :), self%inputs(i, :), &
                    value, grad_x1, grad_x2, hessian, status)
                if (status%code /= FORTNUM_OK) return
                base(i, j) = value
            end do
        end do
        do b = 1, p
            do a = 1, p
                do j = 1, n
                    do i = 1, m
                        out_i = (a - 1)*m + i
                        out_j = (b - 1)*n + j
                        cross_bar(out_i, out_j) = mean_bar(i, a)*self%alpha(out_j)
                    end do
                end do
            end do
        end do
        do j = 1, m
            do k = 1, d
                query_bar(j, k) = 0.0_dp
                do i = 1, n
                    call self%kernel%input_derivatives(query(j, :), self%inputs(i, :), &
                        value, grad_x1, grad_x2, hessian, status)
                    if (status%code /= FORTNUM_OK) return
                    do b = 1, p
                        do a = 1, p
                            out_i = (a - 1)*m + j
                            out_j = (b - 1)*n + i
                            query_bar(j, k) = query_bar(j, k) + cross_bar(out_i, out_j)* &
                                self%coregionalization(a, b)*grad_x1(k)
                        end do
                    end do
                end do
            end do
        end do
        if (any(.not. ieee_is_finite(query_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multi-output GP input VJP: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_output_predict_input_vjp

    subroutine multi_output_predict_parameter_jvp(self, query, direction, mean, mean_dot, status)
        !! Full fitted-state parameter JVP of the posterior mean.
        !!
        !! The packed direction contains kernel coordinates, log noise,
        !! output-major coregionalization weights, and independent variances.
        !! The Cholesky solve is differentiated analytically, so this product
        !! includes both train covariance and cross-covariance effects.
        class(multi_output_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :), direction(:)
        real(dp), intent(out) :: mean(:, :), mean_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: kcross(:, :), kcross_dot(:, :), ktrain(:, :), ktrain_dot(:, :)
        real(dp), allocatable :: cross(:, :), cross_dot(:, :), train_dot(:, :), alpha_dot(:)
        real(dp), allocatable :: b_dot(:, :)
        real(dp), allocatable :: kernel_direction(:)
        integer :: n, m, p, kc, rank, a, b, i, j, out_i, out_j
        integer :: weight_start, independent_start
        real(dp) :: noise_dot

        call multi_output_check_query(self, query, mean, status)
        if (status%code /= FORTNUM_OK) return
        kc = self%kernel%parameter_count()
        if (size(direction) /= self%parameter_count() .or. any(.not. ieee_is_finite(direction)) .or. &
            any(shape(mean_dot) /= shape(mean))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP parameter JVP: direction or output shape is invalid")
            return
        end if
        n = self%n_samples
        m = size(query, 1)
        p = self%n_outputs
        rank = size(self%weights, 2)
        weight_start = kc + 2
        independent_start = weight_start + p*rank
        allocate(kernel_direction(kc), b_dot(p, p))
        if (kc > 0) kernel_direction = direction(:kc)
        call multi_output_coreg_direction(self, direction(weight_start:), b_dot)
        allocate(kcross(n, m), kcross_dot(n, m), ktrain(n, n), ktrain_dot(n, n))
        call self%kernel%matrix_jvp(self%inputs, query, kernel_direction, kcross, kcross_dot, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix_jvp(self%inputs, self%inputs, kernel_direction, ktrain, &
            ktrain_dot, status)
        if (status%code /= FORTNUM_OK) return
        allocate(cross(m*p, n*p), cross_dot(m*p, n*p), train_dot(n*p, n*p), alpha_dot(n*p))
        do b = 1, p
            do a = 1, p
                do j = 1, n
                    do i = 1, m
                        out_i = (a - 1)*m + i
                        out_j = (b - 1)*n + j
                        cross(out_i, out_j) = self%coregionalization(a, b)*kcross(j, i)
                        cross_dot(out_i, out_j) = b_dot(a, b)*kcross(j, i) + &
                            self%coregionalization(a, b)*kcross_dot(j, i)
                    end do
                end do
                do j = 1, n
                    do i = 1, n
                        out_i = (a - 1)*n + i
                        out_j = (b - 1)*n + j
                        train_dot(out_i, out_j) = b_dot(a, b)*ktrain(i, j) + &
                            self%coregionalization(a, b)*ktrain_dot(i, j)
                    end do
                end do
            end do
        end do
        noise_dot = self%noise_variance*direction(kc + 1)
        do i = 1, n*p
            train_dot(i, i) = train_dot(i, i) + noise_dot
        end do
        alpha_dot = -matmul(train_dot, self%alpha)
        call self%factorization%solve(alpha_dot, status)
        if (status%code /= FORTNUM_OK) return
        mean = 0.0_dp
        mean_dot = 0.0_dp
        ! Internal vectors are output-major; transpose each contiguous query
        ! block into the public `(query,output)` array.
        do a = 1, p
            do i = 1, m
                mean(i, a) = dot_product(cross((a - 1)*m + i, :), self%alpha)
                mean_dot(i, a) = dot_product(cross_dot((a - 1)*m + i, :), self%alpha) + &
                    dot_product(cross((a - 1)*m + i, :), alpha_dot)
            end do
        end do
        if (any(.not. ieee_is_finite(mean_dot))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multi-output GP parameter JVP: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_output_predict_parameter_jvp

    subroutine multi_output_predict_parameter_vjp(self, query, mean_bar, parameter_bar, status)
        !! Reverse product of the fitted posterior mean with packed parameters.
        class(multi_output_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :), mean_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: kcross(:, :), ktrain(:, :), cross(:, :), cross_bar(:, :)
        real(dp), allocatable :: alpha_bar(:), lambda(:), matrix_bar(:, :)
        real(dp), allocatable :: kbar_cross(:, :), kbar_train(:, :), b_bar(:, :), local_bar(:)
        integer :: n, m, p, kc, rank, a, b, i, j, out_i, out_j, weight_start, independent_start

        parameter_bar = 0.0_dp
        call multi_output_check_query(self, query, mean_bar, status)
        if (status%code /= FORTNUM_OK) return
        kc = self%kernel%parameter_count()
        if (size(parameter_bar) /= self%parameter_count() .or. any(.not. ieee_is_finite(mean_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP parameter VJP: cotangent or output shape is invalid")
            return
        end if
        n = self%n_samples
        m = size(query, 1)
        p = self%n_outputs
        rank = size(self%weights, 2)
        weight_start = kc + 2
        independent_start = weight_start + p*rank
        allocate(kcross(n, m), ktrain(n, n), cross(m*p, n*p), cross_bar(m*p, n*p))
        call self%kernel%matrix(self%inputs, query, kcross, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix(self%inputs, self%inputs, ktrain, status)
        if (status%code /= FORTNUM_OK) return
        do b = 1, p
            do a = 1, p
                do j = 1, n
                    do i = 1, m
                        out_i = (a - 1)*m + i
                        out_j = (b - 1)*n + j
                        cross(out_i, out_j) = self%coregionalization(a, b)*kcross(j, i)
                        cross_bar(out_i, out_j) = mean_bar(i, a)*self%alpha(out_j)
                    end do
                end do
            end do
        end do
        allocate(alpha_bar(n*p), lambda(n*p), matrix_bar(n*p, n*p))
        alpha_bar = 0.0_dp
        do a = 1, p
            do i = 1, m
                do b = 1, p
                    do j = 1, n
                        alpha_bar((b - 1)*n + j) = alpha_bar((b - 1)*n + j) + &
                            cross((a - 1)*m + i, (b - 1)*n + j)*mean_bar(i, a)
                    end do
                end do
            end do
        end do
        lambda = alpha_bar
        call self%factorization%solve(lambda, status)
        if (status%code /= FORTNUM_OK) return
        matrix_bar = -0.5_dp*(spread(lambda, dim=2, ncopies=n*p)*spread(self%alpha, dim=1, &
            ncopies=n*p) + spread(self%alpha, dim=2, ncopies=n*p)*spread(lambda, dim=1, &
            ncopies=n*p))
        allocate(kbar_cross(n, m), kbar_train(n, n), b_bar(p, p), local_bar(kc))
        kbar_cross = 0.0_dp
        kbar_train = 0.0_dp
        b_bar = 0.0_dp
        do b = 1, p
            do a = 1, p
                do j = 1, n
                    do i = 1, m
                        kbar_cross(j, i) = kbar_cross(j, i) + &
                            self%coregionalization(a, b)*cross_bar((a - 1)*m + i, (b - 1)*n + j)
                    end do
                end do
                do j = 1, n
                    do i = 1, n
                        kbar_train(i, j) = kbar_train(i, j) + &
                            self%coregionalization(a, b)*matrix_bar((a - 1)*n + i, (b - 1)*n + j)
                        b_bar(a, b) = b_bar(a, b) + &
                            matrix_bar((a - 1)*n + i, (b - 1)*n + j)*ktrain(i, j)
                    end do
                end do
                do j = 1, n
                    do i = 1, m
                        b_bar(a, b) = b_bar(a, b) + &
                            cross_bar((a - 1)*m + i, (b - 1)*n + j)*kcross(j, i)
                    end do
                end do
            end do
        end do
        call self%kernel%parameter_vjp(self%inputs, query, kbar_cross, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar(:kc) = local_bar
        call self%kernel%parameter_vjp(self%inputs, self%inputs, kbar_train, local_bar, status)
        if (status%code /= FORTNUM_OK) return
        parameter_bar(:kc) = parameter_bar(:kc) + local_bar
        parameter_bar(kc + 1) = 0.0_dp
        do i = 1, n*p
            parameter_bar(kc + 1) = parameter_bar(kc + 1) + self%noise_variance*matrix_bar(i, i)
        end do
        do a = 1, p
            do rank = 1, size(self%weights, 2)
                parameter_bar(weight_start + (a - 1)*size(self%weights, 2) + rank - 1) = &
                    sum((b_bar(a, :) + b_bar(:, a))*self%weights(:, rank))
            end do
            parameter_bar(independent_start + a - 1) = b_bar(a, a)
        end do
        if (any(.not. ieee_is_finite(parameter_bar))) then
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "multi-output GP parameter VJP: nonfinite product")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_output_predict_parameter_vjp

    subroutine multi_output_coreg_direction(self, direction, b_dot)
        class(multi_output_gp_t), intent(in) :: self
        real(dp), intent(in) :: direction(:)
        real(dp), intent(out) :: b_dot(:, :)
        integer :: p, rank, a, b, ell, index

        p = self%n_outputs
        rank = size(self%weights, 2)
        b_dot = 0.0_dp
        index = 1
        do a = 1, p
            do ell = 1, rank
                do b = 1, p
                    b_dot(a, b) = b_dot(a, b) + direction(index)*self%weights(b, ell)
                    b_dot(b, a) = b_dot(b, a) + direction(index)*self%weights(b, ell)
                end do
                index = index + 1
            end do
        end do
        do a = 1, p
            b_dot(a, a) = b_dot(a, a) + direction(index)
            index = index + 1
        end do
    end subroutine multi_output_coreg_direction

    subroutine multi_output_check_query(self, query, output, status)
        class(multi_output_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :), output(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP product: model is not fitted")
            return
        end if
        if (size(query, 1) < 1 .or. size(query, 2) /= self%kernel%input_dim .or. &
            any(shape(output) /= [size(query, 1), self%n_outputs]) .or. &
            any(.not. ieee_is_finite(query))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP product: query or output shape is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_output_check_query

    subroutine multi_output_check_batch_query(self, query, output, status)
        class(multi_output_gp_t), intent(in) :: self
        real(dp), intent(in) :: query(:, :, :), output(:, :, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP batch product: model is not fitted")
            return
        end if
        if (size(query, 1) < 1 .or. size(query, 2) < 1 .or. &
            size(query, 3) /= self%kernel%input_dim .or. &
            any(shape(output) /= [size(query, 1), size(query, 2), self%n_outputs]) .or. &
            any(.not. ieee_is_finite(query))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP batch product: query or output shape is invalid")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_output_check_batch_query

    subroutine multi_output_predict_batch_device(self, device, query, mean, status)
        !! Batch posterior means through the explicit device contract.
        class(multi_output_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: query(:, :, :)
        real(dp), intent(out) :: mean(:, :, :)
        type(fortnum_status_t), intent(out) :: status

        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_batch(query, mean, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multi-output GP batch prediction: CUDA residency is not implemented")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP batch prediction: device kind is invalid")
        end select
    end subroutine multi_output_predict_batch_device

    subroutine multi_output_predict_batch_input_jvp_device(self, device, query, &
            direction, mean, mean_dot, status)
        !! Batch query-input JVP through the explicit device contract.
        class(multi_output_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: query(:, :, :), direction(:, :, :)
        real(dp), intent(out) :: mean(:, :, :), mean_dot(:, :, :)
        type(fortnum_status_t), intent(out) :: status

        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_batch_input_jvp(query, direction, mean, mean_dot, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multi-output GP batch input JVP: CUDA residency is not implemented")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP batch input JVP: device kind is invalid")
        end select
    end subroutine multi_output_predict_batch_input_jvp_device

    subroutine multi_output_predict_batch_input_vjp_device(self, device, query, &
            mean_bar, query_bar, status)
        !! Batch query-input VJP through the explicit device contract.
        class(multi_output_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: query(:, :, :), mean_bar(:, :, :)
        real(dp), intent(out) :: query_bar(:, :, :)
        type(fortnum_status_t), intent(out) :: status

        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_batch_input_vjp(query, mean_bar, query_bar, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multi-output GP batch input VJP: CUDA residency is not implemented")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP batch input VJP: device kind is invalid")
        end select
    end subroutine multi_output_predict_batch_input_vjp_device

    subroutine multi_output_predict_input_jvp_device(self, device, query, direction, &
            mean, mean_dot, status)
        class(multi_output_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: query(:, :), direction(:, :)
        real(dp), intent(out) :: mean(:, :), mean_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (device%kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multi-output GP input JVP: CUDA residency is not implemented")
            return
        end if
        call self%predict_input_jvp(query, direction, mean, mean_dot, status)
    end subroutine multi_output_predict_input_jvp_device

    subroutine multi_output_predict_input_vjp_device(self, device, query, mean_bar, &
            query_bar, status)
        class(multi_output_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: query(:, :), mean_bar(:, :)
        real(dp), intent(out) :: query_bar(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (device%kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multi-output GP input VJP: CUDA residency is not implemented")
            return
        end if
        call self%predict_input_vjp(query, mean_bar, query_bar, status)
    end subroutine multi_output_predict_input_vjp_device

    subroutine multi_output_predict_parameter_jvp_device(self, device, query, direction, &
            mean, mean_dot, status)
        class(multi_output_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: query(:, :), direction(:)
        real(dp), intent(out) :: mean(:, :), mean_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (device%kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multi-output GP parameter JVP: CUDA residency is not implemented")
            return
        end if
        call self%predict_parameter_jvp(query, direction, mean, mean_dot, status)
    end subroutine multi_output_predict_parameter_jvp_device

    subroutine multi_output_predict_parameter_vjp_device(self, device, query, mean_bar, &
            parameter_bar, status)
        class(multi_output_gp_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: query(:, :), mean_bar(:, :)
        real(dp), intent(out) :: parameter_bar(:)
        type(fortnum_status_t), intent(out) :: status

        if (device%kind /= FORTML_DEVICE_CPU) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "multi-output GP parameter VJP: CUDA residency is not implemented")
            return
        end if
        call self%predict_parameter_vjp(query, mean_bar, parameter_bar, status)
    end subroutine multi_output_predict_parameter_vjp_device

    subroutine multi_output_lml(self, targets, value, status)
        class(multi_output_gp_t), intent(inout) :: self
        real(dp), intent(in) :: targets(:, :)
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: stacked(:)
        real(dp) :: log_determinant
        integer :: n, p, i, j

        value = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP: likelihood before fit")
            return
        end if
        n = self%n_samples
        p = self%n_outputs
        if (size(targets, 1) /= n .or. size(targets, 2) /= p) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "multi-output GP: likelihood target shape is invalid")
            return
        end if
        allocate(stacked(n*p))
        do j = 1, p
            do i = 1, n
                stacked((j - 1)*n + i) = targets(i, j)
            end do
        end do
        call self%factorization%log_determinant(log_determinant, status)
        if (status%code /= FORTNUM_OK) return
        value = -0.5_dp*sum(stacked*self%alpha) - 0.5_dp*log_determinant &
            - 0.5_dp*real(n*p, dp)*log(2.0_dp*PI)
        call status_set(status, FORTNUM_OK, "")
    end subroutine multi_output_lml

end module fortml_multi_output_gp
