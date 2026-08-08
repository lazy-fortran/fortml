module fortml_mlp_last_layer_gp
    !! Finite-feature GP-posterior/NTK initialization for an existing MLP.
    !!
    !! The initializer freezes every hidden layer, augments its feature map
    !! with an intercept, and solves the closed-form kernel-ridge problem
    !! ``(Z^T Z + lambda I) C = Z^T Y``.  It is a deterministic finite-width
    !! approximation to an NNGP/last-layer posterior; it is deliberately not
    !! advertised as an exact infinite-width equivalence.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_mlp, only: mlp_t, MLP_LINEAR
    implicit none
    private

    type, public :: mlp_last_layer_parameter_t
        !! Metadata for one initializer hyperparameter.
        character(len=48) :: name = ""
        real(dp) :: value = 0.0_dp
        real(dp) :: lower_bound = 0.0_dp
        real(dp) :: upper_bound = 0.0_dp
        logical :: trainable = .true.
    end type mlp_last_layer_parameter_t

    type, public :: mlp_last_layer_gp_metadata_t
        !! Explicit state/provenance for the finite-feature approximation.
        character(len=48) :: method = "finite-feature-kernel-ridge"
        character(len=32) :: derivative_scope = "fixed-feature-map"
        integer :: sample_count = 0
        integer :: feature_dimension = 0
        integer :: output_dimension = 0
        real(dp) :: regularization = 0.0_dp
        logical :: exact_infinite_width = .false.
        logical :: cuda_supported = .false.
    end type mlp_last_layer_gp_metadata_t

    type, public :: mlp_last_layer_gp_initializer_t
        private
        real(dp), allocatable :: coefficient_state(:, :)
        real(dp), allocatable :: coefficient_jvp(:, :)
        real(dp) :: regularization_value = 0.0_dp
        integer :: sample_count_value = 0
        integer :: feature_dimension_value = 0
        integer :: output_dimension_value = 0
        logical :: fitted_value = .false.
    contains
        procedure, public :: fit => mlp_last_layer_gp_fit
        procedure, public :: fit_apply => mlp_last_layer_gp_fit_apply
        procedure, public :: apply => mlp_last_layer_gp_apply
        procedure, public :: predict => mlp_last_layer_gp_predict
        procedure, public :: jvp => mlp_last_layer_gp_jvp
        procedure, public :: predict_cuda => mlp_last_layer_gp_predict_cuda
        procedure, public :: apply_cuda => mlp_last_layer_gp_apply_cuda
        procedure, public :: jvp_cuda => mlp_last_layer_gp_jvp_cuda
        procedure, public :: fitted => mlp_last_layer_gp_fitted
        procedure, public :: regularization => mlp_last_layer_gp_regularization
        procedure, public :: sample_count => mlp_last_layer_gp_sample_count
        procedure, public :: feature_dimension => mlp_last_layer_gp_feature_dimension
        procedure, public :: output_dimension => mlp_last_layer_gp_output_dimension
        procedure, public :: metadata => mlp_last_layer_gp_metadata
        procedure, public :: parameter_count => mlp_last_layer_gp_parameter_count
        procedure, public :: parameter_metadata => mlp_last_layer_gp_parameter_metadata
        procedure, public :: parameters => mlp_last_layer_gp_parameters
        procedure, public :: set_parameters => mlp_last_layer_gp_set_parameters
        procedure, public :: coefficients => mlp_last_layer_gp_coefficients
    end type mlp_last_layer_gp_initializer_t

contains

    subroutine mlp_last_layer_gp_fit(self, model, x, target, status, regularization)
        class(mlp_last_layer_gp_initializer_t), intent(inout) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: regularization
        real(dp), allocatable :: hidden(:, :), design(:, :), normal(:, :), rhs(:, :)
        real(dp), allocatable :: coefficient_state(:, :), coefficient_jvp(:, :)
        real(dp) :: lambda
        logical :: solved

        lambda = 1.0e-6_dp
        if (present(regularization)) lambda = regularization
        if (.not. valid_fit_inputs(model, x, target, lambda)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP fit: model, data, or regularization is invalid")
            return
        end if
        call model%feature_map(x, hidden, status)
        if (status%code /= FORTNUM_OK) return
        allocate(design(size(x, 1), size(hidden, 2) + 1))
        design(:, 1:size(hidden, 2)) = hidden
        design(:, size(hidden, 2) + 1) = 1.0_dp
        allocate(normal(size(design, 2), size(design, 2)))
        normal = matmul(transpose(design), design)
        call add_diagonal(normal, lambda)
        allocate(rhs(size(design, 2), size(target, 2)))
        rhs = matmul(transpose(design), target)
        allocate(coefficient_state(size(rhs, 1), size(rhs, 2)))
        call cholesky_solve(normal, rhs, coefficient_state, solved)
        if (.not. solved) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP fit: normal equations are not positive definite")
            return
        end if

        ! dC/dlambda = -(Z^T Z + lambda I)^(-1) C.  Solving once more
        ! avoids storing an inverse and is stable for small positive lambda.
        allocate(coefficient_jvp(size(coefficient_state, 1), &
            size(coefficient_state, 2)))
        call cholesky_solve(normal, coefficient_state, coefficient_jvp, solved)
        if (.not. solved) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP fit: regularization product solve failed")
            return
        end if
        coefficient_jvp = -coefficient_jvp
        if (.not. all(ieee_is_finite(coefficient_state)) .or. &
                .not. all(ieee_is_finite(coefficient_jvp))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP fit: solved coefficients are nonfinite")
            return
        end if

        call move_alloc(coefficient_state, self%coefficient_state)
        call move_alloc(coefficient_jvp, self%coefficient_jvp)
        self%regularization_value = lambda
        self%sample_count_value = size(x, 1)
        self%feature_dimension_value = size(hidden, 2)
        self%output_dimension_value = size(target, 2)
        self%fitted_value = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_last_layer_gp_fit

    subroutine mlp_last_layer_gp_fit_apply(self, model, x, target, status, regularization)
        class(mlp_last_layer_gp_initializer_t), intent(inout) :: self
        class(mlp_t), intent(inout) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: regularization
        real(dp), allocatable :: old_parameters(:), new_parameters(:)
        integer :: old_count

        old_count = model%parameter_count()
        if (old_count < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP fit_apply: model is uninitialized")
            return
        end if
        old_parameters = model%parameters()
        call self%fit(model, x, target, status, regularization)
        if (status%code /= FORTNUM_OK) return
        call self%apply(model, status)
        if (status%code /= FORTNUM_OK) then
            allocate(new_parameters(size(old_parameters)))
            new_parameters = old_parameters
            call model%set_parameters(new_parameters, status)
        end if
    end subroutine mlp_last_layer_gp_fit_apply

    subroutine mlp_last_layer_gp_apply(self, model, status)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status
        integer :: hidden_count, output_count
        real(dp), allocatable :: weight(:, :), bias(:)

        if (.not. self%fitted_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP apply: initializer is unfitted")
            return
        end if
        if (model%parameter_count() < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP apply: model is uninitialized")
            return
        end if
        if (model%output_activation /= MLP_LINEAR) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP apply: final activation is nonlinear")
            return
        end if
        if (.not. allocated(model%layer_sizes)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP apply: model topology is unavailable")
            return
        end if
        if (size(model%layer_sizes) < 3) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP apply: model has no hidden layer")
            return
        end if
        if (model%layer_sizes(size(model%layer_sizes) - 1) /= &
                self%feature_dimension_value .or. &
                model%layer_sizes(size(model%layer_sizes)) /= &
                    self%output_dimension_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP apply: model topology is incompatible")
            return
        end if
        hidden_count = self%feature_dimension_value
        output_count = self%output_dimension_value
        allocate(weight(hidden_count, output_count), bias(output_count))
        weight = self%coefficient_state(1:hidden_count, :)
        bias = self%coefficient_state(hidden_count + 1, :)
        call model%set_last_layer_parameters(weight, bias, status)
    end subroutine mlp_last_layer_gp_apply

    subroutine mlp_last_layer_gp_predict(self, model, x, y, status)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: hidden(:, :), design(:, :)

        if (.not. self%fitted_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP predict: initializer is unfitted")
            return
        end if
        call model%feature_map(x, hidden, status)
        if (status%code /= FORTNUM_OK) return
        if (size(hidden, 2) /= self%feature_dimension_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP predict: feature dimension mismatch")
            return
        end if
        allocate(design(size(x, 1), size(hidden, 2) + 1))
        design(:, 1:size(hidden, 2)) = hidden
        design(:, size(hidden, 2) + 1) = 1.0_dp
        allocate(y(size(x, 1), self%output_dimension_value))
        y = matmul(design, self%coefficient_state)
        if (.not. all(ieee_is_finite(y))) then
            deallocate(y)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP predict: result is nonfinite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_last_layer_gp_predict

    subroutine mlp_last_layer_gp_jvp(self, model, x, regularization_direction, &
            y, dy, status)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), regularization_direction
        real(dp), allocatable, intent(out) :: y(:, :), dy(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: hidden(:, :), design(:, :)

        if (.not. self%fitted_value .or. &
                .not. ieee_is_finite(regularization_direction)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP jvp: initializer or direction is invalid")
            return
        end if
        call model%feature_map(x, hidden, status)
        if (status%code /= FORTNUM_OK) return
        if (size(hidden, 2) /= self%feature_dimension_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP jvp: feature dimension mismatch")
            return
        end if
        allocate(design(size(x, 1), size(hidden, 2) + 1))
        design(:, 1:size(hidden, 2)) = hidden
        design(:, size(hidden, 2) + 1) = 1.0_dp
        allocate(y(size(x, 1), self%output_dimension_value))
        allocate(dy(size(x, 1), self%output_dimension_value))
        y = matmul(design, self%coefficient_state)
        dy = regularization_direction*matmul(design, self%coefficient_jvp)
        if (.not. all(ieee_is_finite(y)) .or. .not. all(ieee_is_finite(dy))) then
            deallocate(y, dy)
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP jvp: result is nonfinite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_last_layer_gp_jvp

    subroutine mlp_last_layer_gp_predict_cuda(self, model, x, y, status)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :)
        real(dp), allocatable, intent(inout) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "last-layer GP predict_cuda: resident CUDA feature map is unavailable")
    end subroutine mlp_last_layer_gp_predict_cuda

    subroutine mlp_last_layer_gp_apply_cuda(self, model, status)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(inout) :: model
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "last-layer GP apply_cuda: resident CUDA state is unavailable")
    end subroutine mlp_last_layer_gp_apply_cuda

    subroutine mlp_last_layer_gp_jvp_cuda(self, model, x, regularization_direction, &
            y, dy, status)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), regularization_direction
        real(dp), allocatable, intent(inout) :: y(:, :), dy(:, :)
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "last-layer GP jvp_cuda: resident CUDA products are unavailable")
    end subroutine mlp_last_layer_gp_jvp_cuda

    logical function mlp_last_layer_gp_fitted(self) result(value)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        value = self%fitted_value
    end function mlp_last_layer_gp_fitted

    real(dp) function mlp_last_layer_gp_regularization(self) result(value)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        value = self%regularization_value
    end function mlp_last_layer_gp_regularization

    integer function mlp_last_layer_gp_sample_count(self) result(value)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        value = self%sample_count_value
    end function mlp_last_layer_gp_sample_count

    integer function mlp_last_layer_gp_feature_dimension(self) result(value)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        value = self%feature_dimension_value
    end function mlp_last_layer_gp_feature_dimension

    integer function mlp_last_layer_gp_output_dimension(self) result(value)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        value = self%output_dimension_value
    end function mlp_last_layer_gp_output_dimension

    function mlp_last_layer_gp_metadata(self) result(metadata)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        type(mlp_last_layer_gp_metadata_t) :: metadata

        metadata%sample_count = self%sample_count_value
        metadata%feature_dimension = self%feature_dimension_value
        metadata%output_dimension = self%output_dimension_value
        metadata%regularization = self%regularization_value
        metadata%exact_infinite_width = .false.
        metadata%cuda_supported = .false.
    end function mlp_last_layer_gp_metadata

    integer function mlp_last_layer_gp_parameter_count(self) result(value)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        value = 1
    end function mlp_last_layer_gp_parameter_count

    function mlp_last_layer_gp_parameter_metadata(self) result(metadata)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        type(mlp_last_layer_parameter_t), allocatable :: metadata(:)

        allocate(metadata(1))
        metadata(1)%name = "regularization"
        metadata(1)%value = self%regularization_value
        metadata(1)%lower_bound = tiny(1.0_dp)
        metadata(1)%upper_bound = huge(1.0_dp)
        metadata(1)%trainable = .true.
    end function mlp_last_layer_gp_parameter_metadata

    function mlp_last_layer_gp_parameters(self) result(values)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        allocate(values(1))
        values(1) = self%regularization_value
    end function mlp_last_layer_gp_parameters

    subroutine mlp_last_layer_gp_set_parameters(self, values, status)
        class(mlp_last_layer_gp_initializer_t), intent(inout) :: self
        real(dp), intent(in) :: values(:)
        type(fortnum_status_t), intent(out) :: status

        if (size(values) /= 1 .or. .not. ieee_is_finite(values(1)) .or. &
                values(1) <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "last-layer GP set_parameters: regularization is invalid")
            return
        end if
        self%regularization_value = values(1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine mlp_last_layer_gp_set_parameters

    function mlp_last_layer_gp_coefficients(self) result(values)
        class(mlp_last_layer_gp_initializer_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%coefficient_state)) then
            allocate(values, source=self%coefficient_state)
        else
            allocate(values(0, 0))
        end if
    end function mlp_last_layer_gp_coefficients

    logical function valid_fit_inputs(model, x, target, lambda) result(valid)
        class(mlp_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :), target(:, :)
        real(dp), intent(in) :: lambda

        valid = ieee_is_finite(lambda) .and. lambda > 0.0_dp
        if (.not. valid) return
        if (size(x, 1) < 1 .or. size(x, 2) < 1 .or. size(target, 1) /= size(x, 1) &
                .or. size(target, 2) < 1) then
            valid = .false.
            return
        end if
        if (model%output_activation /= MLP_LINEAR) then
            valid = .false.
            return
        end if
        valid = all(ieee_is_finite(x)) .and. all(ieee_is_finite(target))
    end function valid_fit_inputs

    subroutine add_diagonal(matrix, value)
        real(dp), intent(inout) :: matrix(:, :)
        real(dp), intent(in) :: value
        integer :: i

        do i = 1, min(size(matrix, 1), size(matrix, 2))
            matrix(i, i) = matrix(i, i) + value
        end do
    end subroutine add_diagonal

    subroutine cholesky_solve(matrix, rhs, solution, solved)
        real(dp), intent(in) :: matrix(:, :), rhs(:, :)
        real(dp), intent(out) :: solution(:, :)
        logical, intent(out) :: solved
        real(dp), allocatable :: lower(:, :), work(:, :)
        real(dp) :: diagonal
        integer :: n, nrhs, i, j, k

        n = size(matrix, 1)
        nrhs = size(rhs, 2)
        solved = .false.
        if (size(matrix, 2) /= n .or. size(rhs, 1) /= n .or. &
                any(shape(solution) /= shape(rhs)) .or. n < 1 .or. nrhs < 1) return
        if (.not. all(ieee_is_finite(matrix)) .or. .not. all(ieee_is_finite(rhs))) return
        allocate(lower(n, n), work(n, nrhs))
        lower = 0.0_dp
        do i = 1, n
            do j = 1, i
                diagonal = matrix(i, j)
                do k = 1, j - 1
                    diagonal = diagonal - lower(i, k)*lower(j, k)
                end do
                if (i == j) then
                    if (.not. ieee_is_finite(diagonal) .or. diagonal <= 0.0_dp) return
                    lower(i, j) = sqrt(diagonal)
                else
                    if (.not. ieee_is_finite(lower(j, j)) .or. lower(j, j) == 0.0_dp) return
                    lower(i, j) = diagonal/lower(j, j)
                end if
            end do
        end do
        work = rhs
        do j = 1, nrhs
            do i = 1, n
                if (i > 1) then
                    work(i, j) = work(i, j) - &
                        sum(lower(i, 1:i - 1)*work(1:i - 1, j))
                end if
                work(i, j) = work(i, j)/lower(i, i)
            end do
            do i = n, 1, -1
                if (i < n) then
                    work(i, j) = work(i, j) - &
                        sum(lower(i + 1:n, i)*work(i + 1:n, j))
                end if
                work(i, j) = work(i, j)/lower(i, i)
            end do
        end do
        if (.not. all(ieee_is_finite(work))) return
        solution = work
        solved = .true.
    end subroutine cholesky_solve

end module fortml_mlp_last_layer_gp
