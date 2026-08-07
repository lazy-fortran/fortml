module fortml_linear_autoencoder
    !! The exact tied linear autoencoder associated with a fitted PCA.
    !!
    !! For a centered data matrix and a requested latent rank, the encoder is
    !! the PCA loading matrix and the decoder is its transpose.  This is the
    !! global minimum of the tied, squared reconstruction objective (up to the
    !! usual component sign/rotation conventions).  The module deliberately
    !! keeps this initializer separate from nonlinear networks: a finite MLP
    !! or an NNGP posterior is not silently claimed to be the same model.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: FORTML_DEVICE_CPU
    use fortml_pca, only: pca_t
    implicit none
    private

    type, public :: linear_autoencoder_t
        !! Centered tied linear encoder/decoder initialized from PCA.
        private
        real(dp), allocatable :: encoder_value(:, :) ! latent x feature
        real(dp), allocatable :: decoder_value(:, :) ! feature x latent
        real(dp), allocatable :: mean_value(:)
        integer :: feature_count_value = 0
        integer :: latent_count_value = 0
    contains
        procedure, public :: fit => linear_autoencoder_fit
        procedure, public :: initialize_from_pca => &
            linear_autoencoder_initialize_from_pca
        procedure, public :: encode => linear_autoencoder_encode
        procedure, public :: decode => linear_autoencoder_decode
        procedure, public :: reconstruct => linear_autoencoder_reconstruct
        procedure, public :: encode_jvp => linear_autoencoder_encode_jvp
        procedure, public :: reconstruct_jvp => linear_autoencoder_reconstruct_jvp
        procedure, public :: encoder_weights => linear_autoencoder_encoder_weights
        procedure, public :: decoder_weights => linear_autoencoder_decoder_weights
        procedure, public :: mean => linear_autoencoder_mean
        procedure, public :: feature_count => linear_autoencoder_feature_count
        procedure, public :: latent_count => linear_autoencoder_latent_count
        procedure, public :: fitted => linear_autoencoder_fitted
        procedure, public :: device_supported => &
            linear_autoencoder_device_supported
    end type linear_autoencoder_t

contains

    subroutine linear_autoencoder_fit(self, x, status, n_components)
        class(linear_autoencoder_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_components
        type(pca_t) :: pca

        if (present(n_components)) then
            call pca%fit(x, status, n_components=n_components)
        else
            call pca%fit(x, status)
        end if
        if (.not. status_ok(status)) return
        call self%initialize_from_pca(pca, status)
    end subroutine linear_autoencoder_fit

    subroutine linear_autoencoder_initialize_from_pca(self, pca, status)
        class(linear_autoencoder_t), intent(out) :: self
        class(pca_t), intent(in) :: pca
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: components(:,:), center(:)

        if (.not. pca%fitted()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear autoencoder: PCA must be fitted")
            return
        end if
        components = pca%components()
        center = pca%mean()
        if (size(components, 1) < 1 .or. size(components, 2) < 1 .or. &
                size(center) /= size(components, 2) .or. &
                any(.not. ieee_is_finite(components)) .or. &
                any(.not. ieee_is_finite(center))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear autoencoder: PCA state is invalid")
            return
        end if
        allocate(self%encoder_value, source=components)
        allocate(self%decoder_value(size(components, 2), size(components, 1)))
        self%decoder_value = transpose(components)
        allocate(self%mean_value, source=center)
        self%feature_count_value = size(components, 2)
        self%latent_count_value = size(components, 1)
        call status_set(status, FORTNUM_OK, "")
    end subroutine linear_autoencoder_initialize_from_pca

    subroutine linear_autoencoder_encode(self, x, latent, status)
        class(linear_autoencoder_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: latent(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: centered(:, :)

        if (.not. valid_encode_shapes(self, x, latent)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear autoencoder encode: model, input, or output shape is invalid")
            return
        end if
        allocate(centered(size(x, 1), size(x, 2)))
        centered = x - spread(self%mean_value, dim=1, ncopies=size(x, 1))
        latent = matmul(centered, transpose(self%encoder_value))
        call status_set(status, FORTNUM_OK, "")
    end subroutine linear_autoencoder_encode

    subroutine linear_autoencoder_decode(self, latent, x, status)
        class(linear_autoencoder_t), intent(in) :: self
        real(dp), intent(in) :: latent(:, :)
        real(dp), intent(out) :: x(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. valid_decode_shapes(self, latent, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear autoencoder decode: model, input, or output shape is invalid")
            return
        end if
        x = matmul(latent, transpose(self%decoder_value)) + &
            spread(self%mean_value, dim=1, ncopies=size(latent, 1))
        call status_set(status, FORTNUM_OK, "")
    end subroutine linear_autoencoder_decode

    subroutine linear_autoencoder_reconstruct(self, x, reconstruction, status)
        class(linear_autoencoder_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: reconstruction(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: latent(:, :)

        if (.not. valid_reconstruct_shapes(self, x, reconstruction)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear autoencoder reconstruct: model, input, or output shape is invalid")
            return
        end if
        allocate(latent(size(x, 1), self%latent_count_value))
        call self%encode(x, latent, status)
        if (.not. status_ok(status)) return
        call self%decode(latent, reconstruction, status)
    end subroutine linear_autoencoder_reconstruct

    subroutine linear_autoencoder_encode_jvp(self, x_dot, latent_dot, status)
        class(linear_autoencoder_t), intent(in) :: self
        real(dp), intent(in) :: x_dot(:, :)
        real(dp), intent(out) :: latent_dot(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. linear_autoencoder_fitted(self) .or. &
                .not. finite_matrix(x_dot) .or. size(x_dot, 2) /= self%feature_count_value .or. &
                any(shape(latent_dot) /= [size(x_dot, 1), self%latent_count_value])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear autoencoder encode_jvp: tangent or output shape is invalid")
            return
        end if
        latent_dot = matmul(x_dot, transpose(self%encoder_value))
        call status_set(status, FORTNUM_OK, "")
    end subroutine linear_autoencoder_encode_jvp

    subroutine linear_autoencoder_reconstruct_jvp(self, x_dot, reconstruction_dot, status)
        class(linear_autoencoder_t), intent(in) :: self
        real(dp), intent(in) :: x_dot(:, :)
        real(dp), intent(out) :: reconstruction_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: latent_dot(:, :)

        if (.not. linear_autoencoder_fitted(self) .or. &
                .not. finite_matrix(x_dot) .or. size(x_dot, 2) /= self%feature_count_value .or. &
                any(shape(reconstruction_dot) /= shape(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "linear autoencoder reconstruct_jvp: tangent or output shape is invalid")
            return
        end if
        allocate(latent_dot(size(x_dot, 1), self%latent_count_value))
        latent_dot = matmul(x_dot, transpose(self%encoder_value))
        reconstruction_dot = matmul(latent_dot, transpose(self%decoder_value))
        call status_set(status, FORTNUM_OK, "")
    end subroutine linear_autoencoder_reconstruct_jvp

    function linear_autoencoder_encoder_weights(self) result(values)
        class(linear_autoencoder_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%encoder_value)) then
            values = self%encoder_value
        else
            allocate(values(0, 0))
        end if
    end function linear_autoencoder_encoder_weights

    function linear_autoencoder_decoder_weights(self) result(values)
        class(linear_autoencoder_t), intent(in) :: self
        real(dp), allocatable :: values(:, :)

        if (allocated(self%decoder_value)) then
            values = self%decoder_value
        else
            allocate(values(0, 0))
        end if
    end function linear_autoencoder_decoder_weights

    function linear_autoencoder_mean(self) result(values)
        class(linear_autoencoder_t), intent(in) :: self
        real(dp), allocatable :: values(:)

        if (allocated(self%mean_value)) then
            values = self%mean_value
        else
            allocate(values(0))
        end if
    end function linear_autoencoder_mean

    integer function linear_autoencoder_feature_count(self) result(count)
        class(linear_autoencoder_t), intent(in) :: self

        count = self%feature_count_value
    end function linear_autoencoder_feature_count

    integer function linear_autoencoder_latent_count(self) result(count)
        class(linear_autoencoder_t), intent(in) :: self

        count = self%latent_count_value
    end function linear_autoencoder_latent_count

    logical function linear_autoencoder_fitted(self) result(value)
        class(linear_autoencoder_t), intent(in) :: self

        value = allocated(self%encoder_value) .and. allocated(self%decoder_value) .and. &
            allocated(self%mean_value) .and. self%feature_count_value > 0 .and. &
            self%latent_count_value > 0
    end function linear_autoencoder_fitted

    logical function linear_autoencoder_device_supported(self, device_kind) result(value)
        class(linear_autoencoder_t), intent(in) :: self
        integer, intent(in) :: device_kind

        ! Matrix products are intentionally host-only until a resident CUDA
        ! lowering is linked; callers must not mistake this for a GPU path.
        value = linear_autoencoder_fitted(self) .and. device_kind == FORTML_DEVICE_CPU
    end function linear_autoencoder_device_supported

    logical function finite_matrix(values) result(value)
        real(dp), intent(in) :: values(:, :)

        value = size(values, 1) > 0 .and. size(values, 2) > 0 .and. &
            all(ieee_is_finite(values))
    end function finite_matrix

    logical function valid_encode_shapes(self, x, latent) result(value)
        class(linear_autoencoder_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), latent(:, :)

        value = linear_autoencoder_fitted(self) .and. finite_matrix(x) .and. &
            size(x, 2) == self%feature_count_value .and. &
            all(shape(latent) == [size(x, 1), self%latent_count_value])
    end function valid_encode_shapes

    logical function valid_decode_shapes(self, latent, x) result(value)
        class(linear_autoencoder_t), intent(in) :: self
        real(dp), intent(in) :: latent(:, :), x(:, :)

        value = linear_autoencoder_fitted(self) .and. finite_matrix(latent) .and. &
            size(latent, 2) == self%latent_count_value .and. &
            all(shape(x) == [size(latent, 1), self%feature_count_value])
    end function valid_decode_shapes

    logical function valid_reconstruct_shapes(self, x, reconstruction) result(value)
        class(linear_autoencoder_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), reconstruction(:, :)

        value = linear_autoencoder_fitted(self) .and. finite_matrix(x) .and. &
            size(x, 2) == self%feature_count_value .and. &
            all(shape(reconstruction) == shape(x))
    end function valid_reconstruct_shapes

end module fortml_linear_autoencoder
