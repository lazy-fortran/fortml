module fortml_sparse_prior_gp
    !! Prior sparse approximations: SoR, DTC, FITC and PITC.
    !!
    !! Following Liu, Ong, Shen and Cai, "When Gaussian Process Meets Big Data"
    !! (IEEE TNNLS 31(11):4405-4423, 2020), Section III-C1, all four modify the
    !! joint prior through the Nystrom term
    !! `Q_nn = K_nm K_mm^{-1} K_mn` and differ only in the residual they keep
    !! and in whether the test conditional stays exact:
    !!
    !!   SoR   residual 0,                     degenerate test conditional
    !!   DTC   residual 0,                     exact test conditional
    !!   FITC  residual diag(K_nn - Q_nn),     exact test conditional
    !!   PITC  residual blkdiag(K_nn - Q_nn),  exact test conditional
    !!
    !! Writing `L = residual + sigma^2 I` and
    !! `S = (K_mm + K_mn L^{-1} K_nm)^{-1}`, every method shares
    !!
    !!     mean(x*) = k_*m S K_mn L^{-1} y
    !!     var(x*)  = k_** - k_*m K_mm^{-1} k_m* + k_*m S k_m*
    !!
    !! with the middle term dropped for SoR, which is exactly why SoR is
    !! overconfident away from the inducing set (paper, Fig. 4).
    !!
    !! Cost is `O(n m^2)` for training and `O(m)` (mean) or `O(m^2)`
    !! (variance) per test point, against `O(n^3)` and `O(n^2)` for exact GP.
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    use fortnum_cholesky, only: cholesky_factorization_t
    use fortml_kernels, only: kernel_t
    implicit none
    private

    integer, parameter, public :: SPARSE_SOR = 1
    integer, parameter, public :: SPARSE_DTC = 2
    integer, parameter, public :: SPARSE_FITC = 3
    integer, parameter, public :: SPARSE_PITC = 4

    real(dp), parameter :: PI = 3.141592653589793238462643_dp

    type, public :: sparse_prior_gp_t
        type(kernel_t) :: kernel
        real(dp), allocatable :: inducing_points(:, :)
        real(dp), allocatable :: inputs(:, :)
        !! `residual` is the kept diagonal of `K_nn - Q_nn` plus the noise; for
        !! PITC the block structure is stored as a dense block list.
        real(dp), allocatable :: residual(:)
        real(dp), allocatable :: residual_blocks(:, :, :)
        real(dp), allocatable :: weights(:)
        type(cholesky_factorization_t) :: prior_factor
        type(cholesky_factorization_t) :: posterior_factor
        real(dp) :: noise_variance = 1.0_dp
        real(dp) :: log_marginal = 0.0_dp
        integer :: method = SPARSE_DTC
        integer :: block_size = 0
        integer :: n_inducing = 0
        integer :: n_samples = 0
        logical :: fitted = .false.
    contains
        procedure, public :: initialize => sparse_prior_initialize
        procedure, public :: fit => sparse_prior_fit
        procedure, public :: predict => sparse_prior_predict
        procedure, public :: log_marginal_likelihood => sparse_prior_lml
    end type sparse_prior_gp_t

    public :: sparse_prior_method_name

contains

    function sparse_prior_method_name(method) result(name)
        integer, intent(in) :: method
        character(len=:), allocatable :: name

        select case (method)
        case (SPARSE_SOR)
            name = "SoR"
        case (SPARSE_DTC)
            name = "DTC"
        case (SPARSE_FITC)
            name = "FITC"
        case (SPARSE_PITC)
            name = "PITC"
        case default
            name = "unknown"
        end select
    end function sparse_prior_method_name

    subroutine sparse_prior_initialize(self, inducing_points, kernel, &
            noise_variance, method, status, block_size)
        class(sparse_prior_gp_t), intent(out) :: self
        real(dp), intent(in) :: inducing_points(:, :)
        type(kernel_t), intent(in) :: kernel
        real(dp), intent(in) :: noise_variance
        integer, intent(in) :: method
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: block_size

        if (size(inducing_points, 1) < 1 .or. noise_variance <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse prior GP: inducing set or noise is invalid")
            return
        end if
        if (size(inducing_points, 2) /= kernel%input_dim) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse prior GP: inducing points do not match the kernel")
            return
        end if
        if (method < SPARSE_SOR .or. method > SPARSE_PITC) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse prior GP: unknown method")
            return
        end if
        self%block_size = 0
        if (method == SPARSE_PITC) then
            if (.not. present(block_size)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "sparse prior GP: PITC needs a block size")
                return
            end if
            if (block_size < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "sparse prior GP: the PITC block size must be positive")
                return
            end if
            self%block_size = block_size
        end if

        self%kernel = kernel
        self%noise_variance = noise_variance
        self%method = method
        self%n_inducing = size(inducing_points, 1)
        allocate(self%inducing_points, source=inducing_points)
        self%fitted = .false.
        call status_set(status, FORTNUM_OK, "")
    end subroutine sparse_prior_initialize

    subroutine sparse_prior_fit(self, x, y, status)
        class(sparse_prior_gp_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: k_uu(:, :), k_uf(:, :), v(:, :)
        real(dp), allocatable :: posterior(:, :), scaled_y(:), rhs(:)
        real(dp) :: jitter, posterior_jitter, quadratic, log_det_residual
        real(dp) :: log_det_ratio
        integer :: i, j, m, n

        self%fitted = .false.
        n = size(x, 1)
        m = self%n_inducing
        if (m < 1 .or. n < 1 .or. size(y) /= n .or. &
            size(x, 2) /= self%kernel%input_dim) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse prior GP: training shape is invalid")
            return
        end if
        if (allocated(self%inputs)) deallocate(self%inputs)
        allocate(self%inputs, source=x)
        self%n_samples = n

        allocate(k_uu(m, m), k_uf(m, n), v(m, n))
        call self%kernel%matrix(self%inducing_points, self%inducing_points, &
            k_uu, status)
        if (status%code /= FORTNUM_OK) return
        call self%kernel%matrix(self%inducing_points, x, k_uf, status)
        if (status%code /= FORTNUM_OK) return
        jitter = 1.0e-10_dp*max(maxval(abs(k_uu)), 1.0_dp)
        do i = 1, m
            k_uu(i, i) = k_uu(i, i) + jitter
        end do
        call self%prior_factor%factorize(k_uu, status)
        if (status%code /= FORTNUM_OK) return
        v = k_uf
        call self%prior_factor%solve(v, status)
        if (status%code /= FORTNUM_OK) return

        call build_residual(self, x, k_uf, v, status)
        if (status%code /= FORTNUM_OK) return

        ! S^{-1} = K_mm + K_mn L^{-1} K_nm.
        allocate(posterior(m, m), scaled_y(n), rhs(m))
        call apply_residual_inverse(self, y, scaled_y)
        posterior = k_uu
        block
            real(dp), allocatable :: scaled_block(:, :)
            integer :: column

            allocate(scaled_block(n, m))
            do column = 1, m
                call apply_residual_inverse(self, k_uf(column, :), &
                    scaled_block(:, column))
            end do
            posterior = posterior + matmul(k_uf, scaled_block)
        end block
        ! The jitter has to be relative to the matrix actually being
        ! factorized. `K_mm + K_mn L^-1 K_nm` grows with the sample count, so a
        ! jitter scaled to `K_mm` alone becomes negligible against it and a
        ! numerically rank-deficient inducing set then fails to factorize at
        ! large `n` while succeeding at small `n`.
        posterior_jitter = 1.0e-12_dp*max(maxval(abs(posterior)), 1.0_dp)
        do i = 1, m
            posterior(i, i) = posterior(i, i) + posterior_jitter
        end do
        call self%posterior_factor%factorize(posterior, status)
        if (status%code /= FORTNUM_OK) return

        rhs = matmul(k_uf, scaled_y)
        if (allocated(self%weights)) deallocate(self%weights)
        allocate(self%weights, source=rhs)
        call self%posterior_factor%solve(self%weights, status)
        if (status%code /= FORTNUM_OK) return

        ! log|Q + L| = log|L| + log|S^{-1}| - log|K_mm| by the matrix
        ! determinant lemma, and y^T (Q + L)^{-1} y by Woodbury.
        quadratic = sum(y*scaled_y) - sum(rhs*self%weights)
        log_det_residual = residual_log_determinant(self)
        block
            real(dp) :: log_det_prior, log_det_posterior

            call self%prior_factor%log_determinant(log_det_prior, status)
            if (status%code /= FORTNUM_OK) return
            call self%posterior_factor%log_determinant(log_det_posterior, status)
            if (status%code /= FORTNUM_OK) return
            log_det_ratio = log_det_posterior - log_det_prior
        end block
        self%log_marginal = -0.5_dp*quadratic &
            - 0.5_dp*(log_det_residual + log_det_ratio) &
            - 0.5_dp*real(n, dp)*log(2.0_dp*PI)
        self%fitted = .true.
        j = 0
    end subroutine sparse_prior_fit

    subroutine build_residual(self, x, k_uf, v, status)
        !! `residual = kept part of K_nn - Q_nn` plus the observation noise.
        class(sparse_prior_gp_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), k_uf(:, :), v(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, n, first, last, local_size, block_index, n_blocks
        real(dp), allocatable :: block(:, :)

        n = size(x, 1)
        if (allocated(self%residual)) deallocate(self%residual)
        if (allocated(self%residual_blocks)) deallocate(self%residual_blocks)
        allocate(self%residual(n))
        self%residual = self%noise_variance

        select case (self%method)
        case (SPARSE_SOR, SPARSE_DTC)
            ! Nothing kept: L = sigma^2 I.
        case (SPARSE_FITC)
            ! `k_ii - q_ii` is non-negative in exact arithmetic, but with many
            ! more data points than inducing points the two terms agree to
            ! within roundoff and the difference can come out slightly
            ! negative. A negative entry makes `L` indefinite and the whole
            ! posterior factorization fails, so the residual is clamped at
            ! zero; the observation noise keeps `L` strictly positive.
            do i = 1, n
                self%residual(i) = self%residual(i) + max( &
                    self%kernel%value(x(i, :), x(i, :)) - sum(v(:, i)*k_uf(:, i)), &
                    0.0_dp)
            end do
        case (SPARSE_PITC)
            n_blocks = (n + self%block_size - 1)/self%block_size
            allocate(self%residual_blocks(self%block_size, self%block_size, &
                n_blocks))
            self%residual_blocks = 0.0_dp
            allocate(block(self%block_size, self%block_size))
            do block_index = 1, n_blocks
                first = (block_index - 1)*self%block_size + 1
                last = min(n, first + self%block_size - 1)
                local_size = last - first + 1
                call self%kernel%matrix(x(first:last, :), x(first:last, :), &
                    block(1:local_size, 1:local_size), status)
                if (status%code /= FORTNUM_OK) return
                do j = 1, local_size
                    do i = 1, local_size
                        block(i, j) = block(i, j) - &
                            sum(v(:, first + i - 1)*k_uf(:, first + j - 1))
                    end do
                    ! Same roundoff floor as FITC, on the block diagonal.
                    block(j, j) = max(block(j, j), 0.0_dp) + self%noise_variance
                end do
                self%residual_blocks(1:local_size, 1:local_size, block_index) = &
                    block(1:local_size, 1:local_size)
            end do
        end select
        call status_set(status, FORTNUM_OK, "")
    end subroutine build_residual

    subroutine apply_residual_inverse(self, input, output)
        !! `output = L^{-1} input`, diagonal or block diagonal.
        class(sparse_prior_gp_t), intent(in) :: self
        real(dp), intent(in) :: input(:)
        real(dp), intent(out) :: output(:)
        real(dp), allocatable :: block(:, :), rhs(:)
        integer :: block_index, first, last, local_size, n_blocks, n

        n = size(input)
        if (self%method /= SPARSE_PITC) then
            output = input/self%residual
            return
        end if
        n_blocks = size(self%residual_blocks, 3)
        allocate(block(self%block_size, self%block_size), rhs(self%block_size))
        do block_index = 1, n_blocks
            first = (block_index - 1)*self%block_size + 1
            last = min(n, first + self%block_size - 1)
            local_size = last - first + 1
            block(1:local_size, 1:local_size) = &
                self%residual_blocks(1:local_size, 1:local_size, block_index)
            rhs(1:local_size) = input(first:last)
            call small_solve(block(1:local_size, 1:local_size), &
                rhs(1:local_size))
            output(first:last) = rhs(1:local_size)
        end do
    end subroutine apply_residual_inverse

    subroutine small_solve(matrix, rhs)
        !! Cholesky solve of one symmetric positive-definite block, in place.
        real(dp), intent(inout) :: matrix(:, :), rhs(:)
        integer :: i, j, k, n
        real(dp) :: total

        n = size(rhs)
        do i = 1, n
            do j = 1, i
                total = matrix(i, j)
                do k = 1, j - 1
                    total = total - matrix(i, k)*matrix(j, k)
                end do
                if (i == j) then
                    matrix(i, i) = sqrt(total)
                else
                    matrix(i, j) = total/matrix(j, j)
                end if
            end do
        end do
        do i = 1, n
            total = rhs(i)
            do k = 1, i - 1
                total = total - matrix(i, k)*rhs(k)
            end do
            rhs(i) = total/matrix(i, i)
        end do
        do i = n, 1, -1
            total = rhs(i)
            do k = i + 1, n
                total = total - matrix(k, i)*rhs(k)
            end do
            rhs(i) = total/matrix(i, i)
        end do
    end subroutine small_solve

    real(dp) function residual_log_determinant(self) result(value)
        class(sparse_prior_gp_t), intent(in) :: self
        real(dp), allocatable :: block(:, :)
        integer :: block_index, first, last, local_size, n_blocks, i, j, k
        real(dp) :: total

        value = 0.0_dp
        if (self%method /= SPARSE_PITC) then
            value = sum(log(self%residual))
            return
        end if
        n_blocks = size(self%residual_blocks, 3)
        allocate(block(self%block_size, self%block_size))
        do block_index = 1, n_blocks
            first = (block_index - 1)*self%block_size + 1
            last = min(self%n_samples, first + self%block_size - 1)
            local_size = last - first + 1
            block(1:local_size, 1:local_size) = &
                self%residual_blocks(1:local_size, 1:local_size, block_index)
            do i = 1, local_size
                do j = 1, i
                    total = block(i, j)
                    do k = 1, j - 1
                        total = total - block(i, k)*block(j, k)
                    end do
                    if (i == j) then
                        block(i, i) = sqrt(total)
                    else
                        block(i, j) = total/block(j, j)
                    end if
                end do
            end do
            do i = 1, local_size
                value = value + 2.0_dp*log(block(i, i))
            end do
        end do
    end function residual_log_determinant

    subroutine sparse_prior_predict(self, query, mean, variance, status)
        class(sparse_prior_gp_t), intent(inout) :: self
        real(dp), intent(in) :: query(:, :)
        real(dp), intent(out) :: mean(:), variance(:)
        type(fortnum_status_t), intent(out) :: status

        real(dp), allocatable :: k_us(:, :), prior_solve(:, :), posterior_solve(:, :)
        integer :: i

        mean = 0.0_dp
        variance = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse prior GP: predict before fit")
            return
        end if
        if (size(query, 2) /= self%kernel%input_dim .or. &
            size(mean) /= size(query, 1) .or. size(variance) /= size(query, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse prior GP: prediction shape is invalid")
            return
        end if

        allocate(k_us(self%n_inducing, size(query, 1)))
        call self%kernel%matrix(self%inducing_points, query, k_us, status)
        if (status%code /= FORTNUM_OK) return
        mean = matmul(transpose(k_us), self%weights)

        allocate(prior_solve, source=k_us)
        allocate(posterior_solve, source=k_us)
        call self%prior_factor%solve(prior_solve, status)
        if (status%code /= FORTNUM_OK) return
        call self%posterior_factor%solve(posterior_solve, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(query, 1)
            variance(i) = sum(k_us(:, i)*posterior_solve(:, i))
            if (self%method /= SPARSE_SOR) then
                ! The exact test conditional keeps k_** - k_*m K_mm^{-1} k_m*.
                variance(i) = variance(i) + &
                    self%kernel%value(query(i, :), query(i, :)) - &
                    sum(k_us(:, i)*prior_solve(:, i))
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine sparse_prior_predict

    subroutine sparse_prior_lml(self, value, status)
        class(sparse_prior_gp_t), intent(in) :: self
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status

        value = 0.0_dp
        if (.not. self%fitted) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "sparse prior GP: likelihood before fit")
            return
        end if
        value = self%log_marginal
        call status_set(status, FORTNUM_OK, "")
    end subroutine sparse_prior_lml

end module fortml_sparse_prior_gp
