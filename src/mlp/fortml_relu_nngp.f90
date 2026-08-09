module fortml_relu_nngp
    !! Exact infinite-width covariance propagation for bias-free ReLU MLPs.
    !!
    !! This module describes the GP limit, not a finite-network weight draw.
    !! With weights distributed as ``N(0, sigma_w^2 / fan_in)`` and independent
    !! biases distributed as ``N(0, sigma_b^2)``, it propagates the covariance
    !! after each ReLU hidden layer using the analytic arc-cosine recurrence.
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    implicit none
    private

    type, public :: relu_nngp_metadata_t
        character(len=40) :: method = "analytic-relu-arc-cosine"
        character(len=40) :: activation = "relu"
        integer :: input_dimension = 0
        integer :: hidden_layer_count = 0
        real(dp) :: weight_variance = 2.0_dp
        real(dp) :: bias_variance = 0.0_dp
        logical :: exact_infinite_width = .true.
        logical :: cuda_supported = .false.
        logical :: finite_mlp_weight_map_supported = .false.
    end type relu_nngp_metadata_t

    type, public :: relu_nngp_t
        private
        integer :: input_dimension_value = 0
        integer :: hidden_layer_count_value = 0
        real(dp) :: weight_variance_value = 2.0_dp
        real(dp) :: bias_variance_value = 0.0_dp
        logical :: configured_value = .false.
    contains
        procedure, public :: configure => relu_nngp_configure
        procedure, public :: covariance => relu_nngp_covariance
        procedure, public :: covariance_cuda => relu_nngp_covariance_cuda
        procedure, public :: configured => relu_nngp_configured
        procedure, public :: metadata => relu_nngp_metadata
    end type relu_nngp_t

contains

    subroutine relu_nngp_configure(self, input_dimension, hidden_layer_count, status, &
            weight_variance, bias_variance)
        class(relu_nngp_t), intent(inout) :: self
        integer, intent(in) :: input_dimension, hidden_layer_count
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: weight_variance, bias_variance
        real(dp) :: sigma_w_squared, sigma_b_squared

        sigma_w_squared = 2.0_dp
        if (present(weight_variance)) sigma_w_squared = weight_variance
        sigma_b_squared = 0.0_dp
        if (present(bias_variance)) sigma_b_squared = bias_variance
        if (input_dimension < 1 .or. hidden_layer_count < 0 .or. &
            .not. ieee_is_finite(sigma_w_squared) .or. &
            .not. ieee_is_finite(sigma_b_squared) .or. &
            sigma_w_squared <= 0.0_dp .or. sigma_b_squared < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ReLU NNGP configure: dimensions or prior variances are invalid")
            return
        end if
        self%input_dimension_value = input_dimension
        self%hidden_layer_count_value = hidden_layer_count
        self%weight_variance_value = sigma_w_squared
        self%bias_variance_value = sigma_b_squared
        self%configured_value = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine relu_nngp_configure

    subroutine relu_nngp_covariance(self, x_left, x_right, value, status)
        class(relu_nngp_t), intent(in) :: self
        real(dp), intent(in) :: x_left(:, :), x_right(:, :)
        real(dp), allocatable, intent(out) :: value(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: current(:, :), left_diagonal(:), right_diagonal(:)
        integer :: layer_index

        if (.not. valid_inputs(self, x_left, x_right)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ReLU NNGP covariance: kernel is unconfigured or inputs are invalid")
            return
        end if
        allocate(current(size(x_left, 1), size(x_right, 1)))
        current = matmul(x_left, transpose(x_right))/real(self%input_dimension_value, dp)
        do layer_index = 1, self%hidden_layer_count_value
            left_diagonal = diagonal_covariance(x_left, self, layer_index - 1)
            right_diagonal = diagonal_covariance(x_right, self, layer_index - 1)
            call relu_covariance_step(current, left_diagonal, right_diagonal, &
                self%weight_variance_value, self%bias_variance_value)
        end do
        if (.not. all(ieee_is_finite(current))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "ReLU NNGP covariance: result is nonfinite")
            return
        end if
        allocate(value, source=current)
        call status_set(status, FORTNUM_OK, "")
    end subroutine relu_nngp_covariance

    subroutine relu_nngp_covariance_cuda(self, x_left, x_right, value, status)
        class(relu_nngp_t), intent(in) :: self
        real(dp), intent(in) :: x_left(:, :), x_right(:, :)
        real(dp), allocatable, intent(inout) :: value(:, :)
        type(fortnum_status_t), intent(out) :: status

        call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
            "ReLU NNGP covariance_cuda: resident CUDA kernel is unavailable")
    end subroutine relu_nngp_covariance_cuda

    logical function relu_nngp_configured(self) result(value)
        class(relu_nngp_t), intent(in) :: self
        value = self%configured_value
    end function relu_nngp_configured

    function relu_nngp_metadata(self) result(value)
        class(relu_nngp_t), intent(in) :: self
        type(relu_nngp_metadata_t) :: value

        value%input_dimension = self%input_dimension_value
        value%hidden_layer_count = self%hidden_layer_count_value
        value%weight_variance = self%weight_variance_value
        value%bias_variance = self%bias_variance_value
    end function relu_nngp_metadata

    logical function valid_inputs(self, x_left, x_right) result(value)
        class(relu_nngp_t), intent(in) :: self
        real(dp), intent(in) :: x_left(:, :), x_right(:, :)

        value = self%configured_value
        if (.not. value) return
        value = size(x_left, 1) > 0 .and. size(x_right, 1) > 0
        if (.not. value) return
        value = size(x_left, 2) == self%input_dimension_value
        if (.not. value) return
        value = size(x_right, 2) == self%input_dimension_value
        if (.not. value) return
        value = all(ieee_is_finite(x_left)) .and. all(ieee_is_finite(x_right))
    end function valid_inputs

    function diagonal_covariance(x, self, propagated_layers) result(value)
        real(dp), intent(in) :: x(:, :)
        class(relu_nngp_t), intent(in) :: self
        integer, intent(in) :: propagated_layers
        real(dp), allocatable :: value(:)
        integer :: layer_index

        allocate(value(size(x, 1)))
        value = sum(x**2, dim=2)/real(self%input_dimension_value, dp)
        do layer_index = 1, propagated_layers
            value = 0.5_dp*self%weight_variance_value*value + self%bias_variance_value
        end do
    end function diagonal_covariance

    subroutine relu_covariance_step(value, left_diagonal, right_diagonal, &
            weight_variance, bias_variance)
        real(dp), intent(inout) :: value(:, :)
        real(dp), intent(in) :: left_diagonal(:), right_diagonal(:)
        real(dp), intent(in) :: weight_variance, bias_variance
        real(dp) :: denominator, correlation, angle
        integer :: i, j

        do j = 1, size(value, 2)
            do i = 1, size(value, 1)
                denominator = sqrt(left_diagonal(i)*right_diagonal(j))
                if (denominator == 0.0_dp) then
                    value(i, j) = bias_variance
                else
                    correlation = max(-1.0_dp, min(1.0_dp, value(i, j)/denominator))
                    angle = acos(correlation)
                    value(i, j) = weight_variance*denominator*(sin(angle) + &
                        (acos(-1.0_dp) - angle)*correlation)/(2.0_dp*acos(-1.0_dp)) + &
                        bias_variance
                end if
            end do
        end do
    end subroutine relu_covariance_step

end module fortml_relu_nngp
