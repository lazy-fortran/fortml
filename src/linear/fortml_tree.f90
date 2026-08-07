!> Deterministic tree primitives and a small gradient-boosting foundation.
module fortml_tree
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    !> A one-level regression tree selected by exhaustive squared-error splits.
    !>
    !> The stump is intentionally deterministic: features are visited in
    !> ascending order and thresholds are midpoints between distinct sorted
    !> observations.  This gives a stable, auditable primitive for the future
    !> CART/XGBoost implementation.  Tree predictions are piecewise constant;
    !> their input derivative is zero away from a split and undefined on it.
    type, public :: decision_stump_t
        private
        integer :: n_inputs = 0
        integer :: feature_index = 0
        integer :: min_samples_leaf = 1
        real(dp) :: threshold = 0.0_dp
        real(dp) :: left_value = 0.0_dp
        real(dp) :: right_value = 0.0_dp
        logical :: initialized = .false.
    contains
        procedure, public :: fit => stump_fit
        procedure, public :: predict_matrix => stump_predict_matrix
        procedure, public :: predict_vector => stump_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: jvp => stump_jvp
        procedure, public :: input_count => stump_input_count
        procedure, public :: split_feature => stump_split_feature
        procedure, public :: split_threshold => stump_split_threshold
        procedure, public :: is_initialized => stump_is_initialized
    end type decision_stump_t

    !> Squared-loss gradient boosting using deterministic regression stumps.
    !>
    !> This is a deliberately small first implementation of the tree-family
    !> API.  It has no differentiable surrogate for split selection: fit is a
    !> discrete operation, while prediction JVPs are available only away from
    !> split boundaries.  Histogram/quantile proposals, subsampling, feature
    !> metadata, and GPU kernels belong to the full tree backend.
    type, public :: gradient_boosting_regressor_t
        private
        integer :: n_inputs = 0
        integer :: n_estimators = 0
        integer :: min_samples_leaf = 1
        real(dp) :: learning_rate = 0.1_dp
        real(dp) :: base_value = 0.0_dp
        type(decision_stump_t), allocatable :: estimators(:)
        logical :: initialized = .false.
    contains
        procedure, public :: fit => boosting_fit
        procedure, public :: predict_matrix => boosting_predict_matrix
        procedure, public :: predict_vector => boosting_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: predict_jvp => boosting_predict_jvp
        procedure, public :: input_count => boosting_input_count
        procedure, public :: estimator_count => boosting_estimator_count
        procedure, public :: is_initialized => boosting_is_initialized
    end type gradient_boosting_regressor_t

contains

    subroutine stump_fit(self, x, y, status, min_samples_leaf)
        class(decision_stump_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: min_samples_leaf
        integer, allocatable :: order(:)
        integer :: n_samples, n_features, j, i, k, leaf_size
        integer :: best_feature
        real(dp) :: best_threshold, best_left, best_right, best_sse
        real(dp) :: sum_left, sum_right, mean_left, mean_right, sse
        real(dp) :: candidate_threshold

        n_samples = size(x, 1)
        n_features = size(x, 2)
        leaf_size = 1
        if (present(min_samples_leaf)) leaf_size = min_samples_leaf
        if (n_samples < 2 .or. n_features < 1 .or. size(y) /= n_samples .or. &
            leaf_size < 1 .or. leaf_size*2 > n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "decision stump fit: invalid dimensions or leaf size")
            return
        end if

        allocate(order(n_samples))
        best_sse = huge(1.0_dp)
        best_feature = 1
        best_threshold = 0.0_dp
        best_left = sum(y)/real(n_samples, dp)
        best_right = best_left

        do j = 1, n_features
            call sort_feature_indices(x(:, j), order)
            do k = leaf_size, n_samples - leaf_size
                if (x(order(k), j) >= x(order(k + 1), j)) cycle
                candidate_threshold = 0.5_dp*(x(order(k), j) + &
                    x(order(k + 1), j))
                sum_left = 0.0_dp
                do i = 1, k
                    sum_left = sum_left + y(order(i))
                end do
                sum_right = sum(y) - sum_left
                mean_left = sum_left/real(k, dp)
                mean_right = sum_right/real(n_samples - k, dp)
                sse = 0.0_dp
                do i = 1, k
                    sse = sse + (y(order(i)) - mean_left)**2
                end do
                do i = k + 1, n_samples
                    sse = sse + (y(order(i)) - mean_right)**2
                end do

                if (sse < best_sse) then
                    best_sse = sse
                    best_feature = j
                    best_threshold = candidate_threshold
                    best_left = mean_left
                    best_right = mean_right
                end if
            end do
        end do

        self%n_inputs = n_features
        self%feature_index = best_feature
        self%min_samples_leaf = leaf_size
        self%threshold = best_threshold
        self%left_value = best_left
        self%right_value = best_right
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine stump_fit

    subroutine stump_predict_matrix(self, x, y, status)
        class(decision_stump_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:,:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(y) /= [size(x, 1), 1])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "decision stump predict: model or array shape is invalid")
            return
        end if
        do i = 1, size(x, 1)
            if (x(i, self%feature_index) < self%threshold) then
                y(i, 1) = self%left_value
            else
                y(i, 1) = self%right_value
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine stump_predict_matrix

    subroutine stump_predict_vector(self, x, y, status)
        class(decision_stump_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: ym(:, :)

        if (size(y) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "decision stump predict: output shape is invalid")
            return
        end if
        allocate(ym(size(y), 1))
        call self%predict_matrix(x, ym, status)
        if (status%code == FORTNUM_OK) y = ym(:, 1)
    end subroutine stump_predict_vector

    !> Product through prediction with respect to x at differentiable points.
    subroutine stump_jvp(self, x, x_dot, y, y_dot, status)
        class(decision_stump_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: y(:), y_dot(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(x_dot) /= shape(x)) .or. size(y) /= size(x, 1) .or. &
            size(y_dot) /= size(y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "decision stump jvp: model or array shape is invalid")
            return
        end if
        do i = 1, size(x, 1)
            if (x(i, self%feature_index) == self%threshold) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "decision stump jvp: derivative is undefined on split")
                return
            end if
        end do
        call self%predict_vector(x, y, status)
        if (status%code /= FORTNUM_OK) return
        y_dot = 0.0_dp
        call status_set(status, FORTNUM_OK, "")
    end subroutine stump_jvp

    integer function stump_input_count(self) result(count)
        class(decision_stump_t), intent(in) :: self
        count = self%n_inputs
    end function stump_input_count

    integer function stump_split_feature(self) result(feature)
        class(decision_stump_t), intent(in) :: self
        feature = self%feature_index
    end function stump_split_feature

    real(dp) function stump_split_threshold(self) result(threshold)
        class(decision_stump_t), intent(in) :: self
        threshold = self%threshold
    end function stump_split_threshold

    logical function stump_is_initialized(self) result(initialized)
        class(decision_stump_t), intent(in) :: self
        initialized = self%initialized
    end function stump_is_initialized

    subroutine boosting_fit(self, x, y, status, n_estimators, learning_rate, &
            min_samples_leaf)
        class(gradient_boosting_regressor_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_estimators, min_samples_leaf
        real(dp), intent(in), optional :: learning_rate
        integer :: n_samples, n_features, n_trees, leaf_size, i
        real(dp) :: rate
        real(dp), allocatable :: prediction(:), residual(:), correction(:)

        n_samples = size(x, 1)
        n_features = size(x, 2)
        n_trees = 20
        if (present(n_estimators)) n_trees = n_estimators
        rate = 0.1_dp
        if (present(learning_rate)) rate = learning_rate
        leaf_size = 1
        if (present(min_samples_leaf)) leaf_size = min_samples_leaf
        if (n_samples < 2 .or. n_features < 1 .or. size(y) /= n_samples .or. &
            n_trees < 1 .or. rate <= 0.0_dp .or. rate > 1.0_dp .or. &
            leaf_size < 1 .or. leaf_size*2 > n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "gradient boosting fit: invalid dimensions or hyperparameters")
            return
        end if

        allocate(self%estimators(n_trees))
        allocate(prediction(n_samples), residual(n_samples), correction(n_samples))
        self%base_value = sum(y)/real(n_samples, dp)
        prediction = self%base_value
        do i = 1, n_trees
            residual = y - prediction
            call self%estimators(i)%fit(x, residual, status, leaf_size)
            if (status%code /= FORTNUM_OK) return
            call self%estimators(i)%predict(x, correction, status)
            if (status%code /= FORTNUM_OK) return
            prediction = prediction + rate*correction
        end do
        self%n_inputs = n_features
        self%n_estimators = n_trees
        self%min_samples_leaf = leaf_size
        self%learning_rate = rate
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine boosting_fit

    subroutine boosting_predict_matrix(self, x, y, status)
        class(gradient_boosting_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: correction(:)
        integer :: i

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(y) /= [size(x, 1), 1])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "gradient boosting predict: model or array shape is invalid")
            return
        end if
        allocate(correction(size(x, 1)))
        y(:, 1) = self%base_value
        do i = 1, self%n_estimators
            call self%estimators(i)%predict(x, correction, status)
            if (status%code /= FORTNUM_OK) return
            y(:, 1) = y(:, 1) + self%learning_rate*correction
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine boosting_predict_matrix

    subroutine boosting_predict_vector(self, x, y, status)
        class(gradient_boosting_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: ym(:, :)

        if (size(y) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "gradient boosting predict: output shape is invalid")
            return
        end if
        allocate(ym(size(y), 1))
        call self%predict_matrix(x, ym, status)
        if (status%code == FORTNUM_OK) y = ym(:, 1)
    end subroutine boosting_predict_vector

    subroutine boosting_predict_jvp(self, x, x_dot, y, y_dot, status)
        class(gradient_boosting_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: y(:), y_dot(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: correction(:), correction_dot(:)
        integer :: i

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(x_dot) /= shape(x)) .or. size(y) /= size(x, 1) .or. &
            size(y_dot) /= size(y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "gradient boosting jvp: model or array shape is invalid")
            return
        end if
        allocate(correction(size(x, 1)), correction_dot(size(x, 1)))
        y = self%base_value
        y_dot = 0.0_dp
        do i = 1, self%n_estimators
            call self%estimators(i)%jvp(x, x_dot, correction, correction_dot, &
                status)
            if (status%code /= FORTNUM_OK) return
            y = y + self%learning_rate*correction
            y_dot = y_dot + self%learning_rate*correction_dot
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine boosting_predict_jvp

    integer function boosting_input_count(self) result(count)
        class(gradient_boosting_regressor_t), intent(in) :: self
        count = self%n_inputs
    end function boosting_input_count

    integer function boosting_estimator_count(self) result(count)
        class(gradient_boosting_regressor_t), intent(in) :: self
        count = self%n_estimators
    end function boosting_estimator_count

    logical function boosting_is_initialized(self) result(initialized)
        class(gradient_boosting_regressor_t), intent(in) :: self
        initialized = self%initialized
    end function boosting_is_initialized

    subroutine sort_feature_indices(values, order)
        real(dp), intent(in) :: values(:)
        integer, intent(out) :: order(:)
        integer :: i, j, key

        do i = 1, size(values)
            order(i) = i
        end do
        do i = 2, size(values)
            key = order(i)
            j = i - 1
            do while (j >= 1)
                if (values(order(j)) <= values(key)) exit
                order(j + 1) = order(j)
                j = j - 1
            end do
            order(j + 1) = key
        end do
    end subroutine sort_feature_indices

end module fortml_tree
