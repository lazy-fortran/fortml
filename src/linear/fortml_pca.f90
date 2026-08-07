module fortml_pca
    !! Centered dense principal-component analysis.
    !!
    !! The fit is a thin LAPACK SVD of the centered sample-by-feature matrix.
    !! Components are rows, as in scikit-learn, and their signs are made
    !! deterministic by making the largest-magnitude loading positive.  The
    !! SVD is a fit-time operation; transform and inverse_transform are affine
    !! maps with exact input JVP/VJP products.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_CONVERGENCE_ERROR
    implicit none
    private

    interface
        subroutine dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, &
                lwork, iwork, info)
            import :: dp
            character(1), intent(in) :: jobz
            integer, intent(in) :: m, n, lda, ldu, ldvt, lwork
            real(dp), intent(inout) :: a(lda, *)
            real(dp), intent(out) :: s(*)
            real(dp), intent(out) :: u(ldu, *), vt(ldvt, *)
            real(dp), intent(inout) :: work(*)
            integer, intent(out) :: iwork(*)
            integer, intent(out) :: info
        end subroutine dgesdd
    end interface

    !> Fitted centered dense PCA estimator.
    type, public :: pca_t
        private
        real(dp), allocatable :: component_values(:, :)
        real(dp), allocatable :: mean_value(:)
        real(dp), allocatable :: singular_value(:)
        real(dp), allocatable :: explained_variance_value(:)
        real(dp), allocatable :: explained_variance_ratio_value(:)
        integer :: feature_count_value = 0
        integer :: component_count_value = 0
        integer :: sample_count_value = 0
        logical :: whiten_value = .false.
    contains
        procedure, public :: fit => pca_fit
        procedure, public :: fit_transform => pca_fit_transform
        procedure, public :: transform => pca_transform
        procedure, public :: inverse_transform => pca_inverse_transform
        procedure, public :: transform_jvp => pca_transform_jvp
        procedure, public :: transform_vjp => pca_transform_vjp
        procedure, public :: components => pca_components
        procedure, public :: mean => pca_mean
        procedure, public :: singular_values => pca_singular_values
        procedure, public :: explained_variance => pca_explained_variance
        procedure, public :: explained_variance_ratio => &
            pca_explained_variance_ratio
        procedure, public :: n_components => pca_n_components
        procedure, public :: feature_count => pca_feature_count
        procedure, public :: sample_count => pca_sample_count
        procedure, public :: whiten => pca_whiten
        procedure, public :: fitted => pca_fitted
    end type pca_t

contains

    !> Fit centered PCA with one through min(n_samples,n_features) components.
    subroutine pca_fit(self, x, status, n_components, whiten)
        class(pca_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_components
        logical, intent(in), optional :: whiten
        real(dp), allocatable :: centered(:, :), u(:, :), vt(:, :), s(:), work(:)
        real(dp) :: work_query(1), total_variance
        integer, allocatable :: iwork(:)
        integer :: n_samples, n_features, min_dimension, requested, lwork
        integer :: info, i, j, pivot
        logical :: selected_whiten

        if (.not. valid_input_matrix(x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pca fit: input must be a finite matrix with at least two rows")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        min_dimension = min(n_samples, n_features)
        requested = min_dimension
        if (present(n_components)) requested = n_components
        if (requested < 1 .or. requested > min_dimension) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pca fit: n_components must be in [1,min(n_samples,n_features)]")
            return
        end if
        selected_whiten = .false.
        if (present(whiten)) selected_whiten = whiten

        allocate(centered(n_samples, n_features), self%mean_value(n_features))
        self%mean_value = sum(x, dim=1) / real(n_samples, dp)
        centered = x - spread(self%mean_value, dim=1, ncopies=n_samples)

        allocate(s(min_dimension), u(n_samples, min_dimension))
        allocate(vt(min_dimension, n_features), iwork(8*min_dimension))
        lwork = -1
        call dgesdd("S", n_samples, n_features, centered, n_samples, s, u, &
            n_samples, vt, min_dimension, work_query, lwork, iwork, info)
        if (info /= 0) then
            call invalidate_pca(self)
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "pca fit: SVD workspace query failed")
            return
        end if
        lwork = max(1, int(work_query(1)))
        allocate(work(lwork))
        call dgesdd("S", n_samples, n_features, centered, n_samples, s, u, &
            n_samples, vt, min_dimension, work, lwork, iwork, info)
        if (info /= 0) then
            call invalidate_pca(self)
            call status_set(status, FORTNUM_CONVERGENCE_ERROR, &
                "pca fit: centered SVD failed")
            return
        end if

        allocate(self%component_values(requested, n_features))
        allocate(self%singular_value(requested))
        allocate(self%explained_variance_value(requested))
        allocate(self%explained_variance_ratio_value(requested))
        self%component_values = vt(1:requested, :)
        self%singular_value = s(1:requested)
        do i = 1, requested
            pivot = 1
            do j = 2, n_features
                if (abs(self%component_values(i, j)) > &
                    abs(self%component_values(i, pivot))) pivot = j
            end do
            if (self%component_values(i, pivot) < 0.0_dp) then
                self%component_values(i, :) = -self%component_values(i, :)
            end if
        end do
        self%explained_variance_value = self%singular_value**2 / &
            real(n_samples - 1, dp)
        total_variance = sum(s**2) / real(n_samples - 1, dp)
        if (total_variance > 0.0_dp) then
            self%explained_variance_ratio_value = &
                self%explained_variance_value / total_variance
        else
            self%explained_variance_ratio_value = 0.0_dp
        end if
        self%feature_count_value = n_features
        self%component_count_value = requested
        self%sample_count_value = n_samples
        self%whiten_value = selected_whiten
        call status_set(status, FORTNUM_OK, "")
    end subroutine pca_fit

    !> Fit and immediately project the training matrix.
    subroutine pca_fit_transform(self, x, transformed, status, n_components, &
            whiten)
        class(pca_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_components
        logical, intent(in), optional :: whiten

        call self%fit(x, status, n_components, whiten)
        if (status%code /= FORTNUM_OK) return
        call self%transform(x, transformed, status)
    end subroutine pca_fit_transform

    !> Project rows onto the fitted centered principal subspace.
    subroutine pca_transform(self, x, transformed, status)
        class(pca_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: transformed(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: centered(:, :)
        integer :: j

        if (.not. valid_transform_shapes(self, x, transformed)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pca transform: model, input, or output shape is invalid")
            return
        end if
        allocate(centered(size(x, 1), size(x, 2)))
        centered = x - spread(self%mean_value, dim=1, ncopies=size(x, 1))
        transformed = matmul(centered, transpose(self%component_values))
        if (self%whiten_value) then
            do j = 1, self%component_count_value
                if (self%explained_variance_value(j) > tiny(1.0_dp)) then
                    transformed(:, j) = transformed(:, j) / &
                        sqrt(self%explained_variance_value(j))
                end if
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine pca_transform

    !> Map principal coordinates back to the original feature space.
    subroutine pca_inverse_transform(self, transformed, x, status)
        class(pca_t), intent(in) :: self
        real(dp), intent(in) :: transformed(:, :)
        real(dp), intent(out) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: coordinates(:, :)
        integer :: j

        if (.not. valid_inverse_shapes(self, transformed, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pca inverse_transform: model, input, or output shape is invalid")
            return
        end if
        allocate(coordinates(size(transformed, 1), size(transformed, 2)))
        coordinates = transformed
        if (self%whiten_value) then
            do j = 1, self%component_count_value
                if (self%explained_variance_value(j) > tiny(1.0_dp)) then
                    coordinates(:, j) = coordinates(:, j) * &
                        sqrt(self%explained_variance_value(j))
                end if
            end do
        end if
        x = matmul(coordinates, self%component_values)
        x = x + spread(self%mean_value, dim=1, ncopies=size(transformed, 1))
        call status_set(status, FORTNUM_OK, "")
    end subroutine pca_inverse_transform

    !> Exact input JVP of transform for a fixed fitted SVD state.
    subroutine pca_transform_jvp(self, x_dot, transformed_dot, status)
        class(pca_t), intent(in) :: self
        real(dp), intent(in) :: x_dot(:, :)
        real(dp), intent(out) :: transformed_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: j

        if (.not. pca_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pca transform_jvp: model is not fitted")
            return
        end if
        if (.not. valid_finite_matrix(x_dot)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pca transform_jvp: tangent must be finite and nonempty")
            return
        end if
        if (size(x_dot, 2) /= self%feature_count_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pca transform_jvp: tangent feature count is invalid")
            return
        end if
        if (any(shape(transformed_dot) /= [size(x_dot, 1), &
            self%component_count_value])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pca transform_jvp: output shape is invalid")
            return
        end if
        transformed_dot = matmul(x_dot, transpose(self%component_values))
        if (self%whiten_value) then
            do j = 1, self%component_count_value
                if (self%explained_variance_value(j) > tiny(1.0_dp)) then
                    transformed_dot(:, j) = transformed_dot(:, j) / &
                        sqrt(self%explained_variance_value(j))
                end if
            end do
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine pca_transform_jvp

    !> Exact input VJP of transform for a fixed fitted SVD state.
    subroutine pca_transform_vjp(self, transformed_bar, x_bar, status)
        class(pca_t), intent(in) :: self
        real(dp), intent(in) :: transformed_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: scaled_bar(:, :)
        integer :: j

        if (.not. pca_fitted(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pca transform_vjp: model is not fitted")
            return
        end if
        if (.not. valid_finite_matrix(transformed_bar)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pca transform_vjp: cotangent must be finite and nonempty")
            return
        end if
        if (size(transformed_bar, 2) /= self%component_count_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pca transform_vjp: cotangent component count is invalid")
            return
        end if
        if (any(shape(x_bar) /= [size(transformed_bar, 1), &
            self%feature_count_value])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "pca transform_vjp: output shape is invalid")
            return
        end if
        allocate(scaled_bar(size(transformed_bar, 1), size(transformed_bar, 2)))
        scaled_bar = transformed_bar
        if (self%whiten_value) then
            do j = 1, self%component_count_value
                if (self%explained_variance_value(j) > tiny(1.0_dp)) then
                    scaled_bar(:, j) = scaled_bar(:, j) / &
                        sqrt(self%explained_variance_value(j))
                end if
            end do
        end if
        x_bar = matmul(scaled_bar, self%component_values)
        call status_set(status, FORTNUM_OK, "")
    end subroutine pca_transform_vjp

    function pca_components(self) result(values)
        class(pca_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%component_values)) then
            values = self%component_values
        else
            allocate(values(0, 0))
        end if
    end function pca_components

    function pca_mean(self) result(values)
        class(pca_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%mean_value)) then
            values = self%mean_value
        else
            allocate(values(0))
        end if
    end function pca_mean

    function pca_singular_values(self) result(values)
        class(pca_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%singular_value)) then
            values = self%singular_value
        else
            allocate(values(0))
        end if
    end function pca_singular_values

    function pca_explained_variance(self) result(values)
        class(pca_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%explained_variance_value)) then
            values = self%explained_variance_value
        else
            allocate(values(0))
        end if
    end function pca_explained_variance

    function pca_explained_variance_ratio(self) result(values)
        class(pca_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%explained_variance_ratio_value)) then
            values = self%explained_variance_ratio_value
        else
            allocate(values(0))
        end if
    end function pca_explained_variance_ratio

    integer function pca_n_components(self) result(count)
        class(pca_t), intent(in) :: self

        count = self%component_count_value
    end function pca_n_components

    integer function pca_feature_count(self) result(count)
        class(pca_t), intent(in) :: self

        count = self%feature_count_value
    end function pca_feature_count

    integer function pca_sample_count(self) result(count)
        class(pca_t), intent(in) :: self

        count = self%sample_count_value
    end function pca_sample_count

    logical function pca_whiten(self) result(value)
        class(pca_t), intent(in) :: self

        value = self%whiten_value
    end function pca_whiten

    logical function pca_fitted(self) result(value)
        class(pca_t), intent(in) :: self

        value = allocated(self%component_values)
        if (.not. value) return
        value = allocated(self%mean_value)
        if (.not. value) return
        value = self%feature_count_value > 0 .and. self%component_count_value > 0
    end function pca_fitted

    logical function valid_input_matrix(x) result(valid)
        real(dp), intent(in) :: x(:, :)

        valid = valid_finite_matrix(x)
        if (.not. valid) return
        valid = size(x, 1) >= 2
    end function valid_input_matrix

    logical function valid_finite_matrix(x) result(valid)
        real(dp), intent(in) :: x(:, :)

        valid = size(x, 1) > 0
        if (.not. valid) return
        valid = size(x, 2) > 0
        if (.not. valid) return
        valid = all(ieee_is_finite(x))
    end function valid_finite_matrix

    logical function valid_transform_shapes(self, x, transformed) result(valid)
        class(pca_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), transformed(:, :)

        valid = pca_fitted(self)
        if (.not. valid) return
        valid = valid_finite_matrix(x)
        if (.not. valid) return
        valid = size(x, 2) == self%feature_count_value
        if (.not. valid) return
        valid = all(shape(transformed) == [size(x, 1), self%component_count_value])
    end function valid_transform_shapes

    logical function valid_inverse_shapes(self, transformed, x) result(valid)
        class(pca_t), intent(in) :: self
        real(dp), intent(in) :: transformed(:, :), x(:, :)

        valid = pca_fitted(self)
        if (.not. valid) return
        valid = valid_finite_matrix(transformed)
        if (.not. valid) return
        valid = size(transformed, 2) == self%component_count_value
        if (.not. valid) return
        valid = all(shape(x) == [size(transformed, 1), self%feature_count_value])
    end function valid_inverse_shapes

    subroutine invalidate_pca(self)
        class(pca_t), intent(inout) :: self

        if (allocated(self%component_values)) deallocate(self%component_values)
        if (allocated(self%mean_value)) deallocate(self%mean_value)
        if (allocated(self%singular_value)) deallocate(self%singular_value)
        if (allocated(self%explained_variance_value)) &
            deallocate(self%explained_variance_value)
        if (allocated(self%explained_variance_ratio_value)) &
            deallocate(self%explained_variance_ratio_value)
        self%feature_count_value = 0
        self%component_count_value = 0
        self%sample_count_value = 0
        self%whiten_value = .false.
    end subroutine invalidate_pca

end module fortml_pca
