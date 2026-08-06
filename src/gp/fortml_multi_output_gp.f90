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
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortml_kernels, only: kernel_t
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
