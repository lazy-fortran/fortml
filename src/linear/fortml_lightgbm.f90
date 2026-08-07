!> A bounded LightGBM-style leaf-wise histogram boosting estimator.
module fortml_lightgbm
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgb_histogram_cut_positions
    implicit none
    private

    integer, parameter, public :: LIGHTGBM_OBJECTIVE_REGRESSION = 1
    integer, parameter, public :: LIGHTGBM_OBJECTIVE_BINARY = 2

    !> Options for the deterministic numeric LightGBM-style path.  The
    !> estimator intentionally exposes only the supported CPU core: weighted
    !> squared regression and binary logistic loss, weighted-quantile
    !> histograms, and best-first (leaf-wise) growth up to `num_leaves`.
    type, public :: lightgbm_options_t
        integer :: n_estimators = 50
        integer :: num_leaves = 31
        integer :: max_depth = 0
        integer :: min_data_in_leaf = 20
        integer :: max_bin = 255
        real(dp) :: learning_rate = 0.1_dp
        real(dp) :: l2 = 1.0_dp
        real(dp) :: min_gain_to_split = 0.0_dp
        character(len=32) :: objective = "regression"
    end type lightgbm_options_t

    type :: lgbm_node_t
        integer :: feature = 0
        integer :: left_child = 0
        integer :: right_child = 0
        integer :: depth = 0
        real(dp) :: threshold = 0.0_dp
        real(dp) :: weight = 0.0_dp
        real(dp) :: gain = 0.0_dp
        logical :: leaf = .true.
        integer, allocatable :: rows(:)
    end type lgbm_node_t

    type :: lgbm_tree_t
        integer :: n_nodes = 0
        integer :: depth = 0
        type(lgbm_node_t), allocatable :: node(:)
    end type lgbm_tree_t

    type :: lgbm_split_t
        logical :: valid = .false.
        integer :: feature = 0
        integer :: node = 0
        real(dp) :: threshold = 0.0_dp
        real(dp) :: gain = 0.0_dp
        real(dp) :: left_weight = 0.0_dp
        real(dp) :: right_weight = 0.0_dp
        integer, allocatable :: left_rows(:), right_rows(:)
    end type lgbm_split_t

    !> Numeric LightGBM-style estimator.  Unlike `xgboost_t`, which grows
    !> each tree recursively by depth, this policy evaluates all current leaf
    !> candidates and splits the globally best leaf until `num_leaves` is
    !> reached.  It is deliberately separate so the XGBoost API and growth
    !> semantics remain unchanged.
    type, public :: lightgbm_t
        private
        integer :: n_inputs = 0
        integer :: objective_code = 0
        integer :: n_estimators = 0
        integer :: num_leaves_value = 0
        integer :: max_bin_value = 0
        integer :: max_depth_value = 0
        integer :: min_data_in_leaf_value = 0
        real(dp) :: learning_rate = 0.0_dp
        real(dp) :: l2_value = 0.0_dp
        real(dp) :: min_gain_value = 0.0_dp
        real(dp) :: base_score = 0.0_dp
        type(lgbm_tree_t), allocatable :: estimator(:)
        logical :: initialized = .false.
    contains
        procedure, public :: fit => lgbm_fit
        procedure, public :: fit_regression => lgbm_fit_regression
        procedure, public :: fit_binary => lgbm_fit_binary
        procedure, public :: predict => lgbm_predict
        procedure, public :: predict_margin => lgbm_predict_margin
        procedure, public :: predict_proba => lgbm_predict_proba
        procedure, public :: predict_device => lgbm_predict_device
        procedure, public :: predict_jvp => lgbm_predict_jvp
        procedure, public :: predict_vjp => lgbm_predict_vjp
        procedure, public :: device_supported => lgbm_device_supported
        procedure, public :: fitted => lgbm_fitted
        procedure, public :: objective_name => lgbm_objective_name
        procedure, public :: estimator_count => lgbm_estimator_count
        procedure, public :: num_leaves => lgbm_num_leaves
        procedure, public :: tree_node_count => lgbm_tree_node_count
        procedure, public :: tree_depth => lgbm_tree_depth
        procedure, public :: feature_count => lgbm_feature_count
        procedure, public :: base_margin => lgbm_base_margin
    end type lightgbm_t

contains

    subroutine lgbm_fit_regression(self, x, y, status, options, sample_weight)
        class(lightgbm_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(lightgbm_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        type(lightgbm_options_t) :: settings

        settings = lightgbm_options_t()
        if (present(options)) settings = options
        settings%objective = "regression"
        call lgbm_fit(self, x, y, status, settings, sample_weight)
    end subroutine lgbm_fit_regression

    subroutine lgbm_fit_binary(self, x, y, status, options, sample_weight)
        class(lightgbm_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(lightgbm_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        type(lightgbm_options_t) :: settings

        settings = lightgbm_options_t()
        if (present(options)) settings = options
        settings%objective = "binary"
        call lgbm_fit(self, x, y, status, settings, sample_weight)
    end subroutine lgbm_fit_binary

    subroutine lgbm_fit(self, x, y, status, options, sample_weight)
        class(lightgbm_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(lightgbm_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        type(lightgbm_options_t) :: settings
        real(dp), allocatable :: weights(:), margin(:), gradient(:), hessian(:), correction(:)
        integer, allocatable :: rows(:)
        integer :: objective_code, i, n_samples, n_features
        real(dp) :: weight_sum, mean_target

        settings = lightgbm_options_t()
        if (present(options)) settings = options
        objective_code = parse_lgbm_objective(settings%objective)
        if (objective_code == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm fit: objective must be regression or binary")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (n_samples < 2 .or. n_features < 1 .or. size(y) /= n_samples .or. &
            settings%n_estimators < 1 .or. settings%num_leaves < 2 .or. &
            settings%num_leaves > n_samples .or. settings%max_depth < 0 .or. &
            settings%min_data_in_leaf < 1 .or. &
            2*settings%min_data_in_leaf > n_samples .or. settings%max_bin < 2 .or. &
            .not. ieee_is_finite(settings%learning_rate) .or. &
            settings%learning_rate <= 0.0_dp .or. settings%learning_rate > 1.0_dp .or. &
            .not. ieee_is_finite(settings%l2) .or. settings%l2 < 0.0_dp .or. &
            .not. ieee_is_finite(settings%min_gain_to_split) .or. &
            settings%min_gain_to_split < 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm fit: invalid dimensions or options")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm fit: only finite numeric inputs are supported")
            return
        end if
        if (objective_code == LIGHTGBM_OBJECTIVE_BINARY) then
            if (any(y < 0.0_dp) .or. any(y > 1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "lightgbm fit: binary targets must lie in [0,1]")
                return
            end if
        end if
        allocate(weights(n_samples))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "lightgbm fit: sample weights must be finite and positive")
                return
            end if
            weights = sample_weight
        end if
        weight_sum = sum(weights)
        mean_target = sum(weights*y)/weight_sum
        if (objective_code == LIGHTGBM_OBJECTIVE_BINARY) then
            self%base_score = stable_logit(mean_target)
        else
            self%base_score = mean_target
        end if
        allocate(self%estimator(settings%n_estimators), margin(n_samples), &
            gradient(n_samples), hessian(n_samples), correction(n_samples), rows(n_samples))
        rows = [(i, i=1,n_samples)]
        margin = self%base_score
        do i = 1, settings%n_estimators
            call lgbm_objective_derivatives(objective_code, margin, y, weights, gradient, &
                hessian, status)
            if (status%code /= FORTNUM_OK) return
            call build_leafwise_tree(x, gradient, hessian, weights, settings, rows, &
                self%estimator(i), status)
            if (status%code /= FORTNUM_OK) return
            call lgbm_tree_predict(self%estimator(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            margin = margin + settings%learning_rate*correction
        end do
        self%n_inputs = n_features
        self%objective_code = objective_code
        self%n_estimators = settings%n_estimators
        self%num_leaves_value = settings%num_leaves
        self%max_bin_value = settings%max_bin
        self%max_depth_value = settings%max_depth
        self%min_data_in_leaf_value = settings%min_data_in_leaf
        self%learning_rate = settings%learning_rate
        self%l2_value = settings%l2
        self%min_gain_value = settings%min_gain_to_split
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_fit

    subroutine lgbm_predict(self, x, y, status)
        class(lightgbm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status

        call lgbm_predict_margin(self, x, y, status)
        if (status%code /= FORTNUM_OK) return
        if (self%objective_code == LIGHTGBM_OBJECTIVE_BINARY) y = stable_sigmoid_array(y)
    end subroutine lgbm_predict

    subroutine lgbm_predict_margin(self, x, margin, status)
        class(lightgbm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margin(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: correction(:)
        integer :: i

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(margin) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict: model, shape, or finite-input contract failed")
            return
        end if
        allocate(correction(size(x, 1)))
        margin = self%base_score
        do i = 1, self%n_estimators
            call lgbm_tree_predict(self%estimator(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            margin = margin + self%learning_rate*correction
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_predict_margin

    subroutine lgbm_predict_proba(self, x, probabilities, status)
        class(lightgbm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margin(:)

        if (self%objective_code /= LIGHTGBM_OBJECTIVE_BINARY) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_proba: only binary objective has probabilities")
            return
        end if
        if (size(probabilities, 1) /= size(x, 1) .or. size(probabilities, 2) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_proba: output shape is invalid")
            return
        end if
        allocate(margin(size(x, 1)))
        call lgbm_predict_margin(self, x, margin, status)
        if (status%code /= FORTNUM_OK) return
        probabilities(:, 2) = stable_sigmoid_array(margin)
        probabilities(:, 1) = 1.0_dp - probabilities(:, 2)
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_predict_proba

    subroutine lgbm_predict_device(self, device, x, y, status)
        class(lightgbm_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, y, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "lightgbm device prediction: no resident CUDA histogram kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm device prediction: device kind is invalid")
        end select
    end subroutine lgbm_predict_device

    subroutine lgbm_predict_jvp(self, x, x_dot, y, y_dot, status)
        class(lightgbm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: y(:), y_dot(:)
        type(fortnum_status_t), intent(out) :: status

        if (size(x_dot, 1) /= size(x, 1) .or. size(x_dot, 2) /= size(x, 2) .or. &
            size(y) /= size(x, 1) .or. size(y_dot) /= size(y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_jvp: shape is invalid")
            return
        end if
        call self%predict(x, y, status)
        if (status%code /= FORTNUM_OK) return
        y_dot = 0.0_dp
        if (leaf_boundary_hit(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_jvp: split-boundary derivative is undefined")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_predict_jvp

    subroutine lgbm_predict_vjp(self, x, output_bar, x_bar, status)
        class(lightgbm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: y(:)

        if (size(output_bar) /= size(x, 1) .or. size(x_bar, 1) /= size(x, 1) .or. &
            size(x_bar, 2) /= size(x, 2) .or. any(.not. ieee_is_finite(output_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_vjp: shape or cotangent is invalid")
            return
        end if
        allocate(y(size(x, 1)))
        call self%predict(x, y, status)
        if (status%code /= FORTNUM_OK) return
        x_bar = 0.0_dp
        if (leaf_boundary_hit(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "lightgbm predict_vjp: split-boundary derivative is undefined")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_predict_vjp

    logical function lgbm_device_supported(self, device_kind) result(supported)
        class(lightgbm_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = self%initialized .and. device_kind == FORTML_DEVICE_CPU
    end function lgbm_device_supported

    logical function lgbm_fitted(self) result(value)
        class(lightgbm_t), intent(in) :: self
        value = self%initialized
    end function lgbm_fitted

    character(len=16) function lgbm_objective_name(self) result(name)
        class(lightgbm_t), intent(in) :: self
        if (self%objective_code == LIGHTGBM_OBJECTIVE_BINARY) then
            name = "binary"
        else if (self%objective_code == LIGHTGBM_OBJECTIVE_REGRESSION) then
            name = "regression"
        else
            name = "unfitted"
        end if
    end function lgbm_objective_name

    integer function lgbm_estimator_count(self) result(value)
        class(lightgbm_t), intent(in) :: self
        value = self%n_estimators
    end function lgbm_estimator_count

    integer function lgbm_num_leaves(self) result(value)
        class(lightgbm_t), intent(in) :: self
        value = self%num_leaves_value
    end function lgbm_num_leaves

    integer function lgbm_tree_node_count(self, tree_index) result(value)
        class(lightgbm_t), intent(in) :: self
        integer, intent(in) :: tree_index
        value = 0
        if (.not. self%initialized .or. tree_index < 1 .or. tree_index > self%n_estimators) return
        value = self%estimator(tree_index)%n_nodes
    end function lgbm_tree_node_count

    integer function lgbm_tree_depth(self, tree_index) result(value)
        class(lightgbm_t), intent(in) :: self
        integer, intent(in) :: tree_index
        value = 0
        if (.not. self%initialized .or. tree_index < 1 .or. tree_index > self%n_estimators) return
        value = self%estimator(tree_index)%depth
    end function lgbm_tree_depth

    integer function lgbm_feature_count(self) result(value)
        class(lightgbm_t), intent(in) :: self
        value = self%n_inputs
    end function lgbm_feature_count

    real(dp) function lgbm_base_margin(self) result(value)
        class(lightgbm_t), intent(in) :: self
        value = self%base_score
    end function lgbm_base_margin

    subroutine lgbm_objective_derivatives(code, margin, target, weights, gradient, hessian, status)
        integer, intent(in) :: code
        real(dp), intent(in) :: margin(:), target(:), weights(:)
        real(dp), intent(out) :: gradient(:), hessian(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: p
        integer :: i

        if (size(margin) /= size(target) .or. size(weights) /= size(target) .or. &
            size(gradient) /= size(target) .or. size(hessian) /= size(target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "lightgbm derivatives: shape is invalid")
            return
        end if
        if (code == LIGHTGBM_OBJECTIVE_REGRESSION) then
            gradient = weights*(margin-target)
            hessian = weights
        else if (code == LIGHTGBM_OBJECTIVE_BINARY) then
            do i = 1, size(target)
                p = stable_sigmoid(margin(i))
                gradient(i) = weights(i)*(p-target(i))
                hessian(i) = weights(i)*max(p*(1.0_dp-p), 1.0e-12_dp)
            end do
        else
            call status_set(status, FORTNUM_DOMAIN_ERROR, "lightgbm derivatives: objective is unsupported")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_objective_derivatives

    subroutine build_leafwise_tree(x, gradient, hessian, observation_weight, options, rows, tree, status)
        real(dp), intent(in) :: x(:, :), gradient(:), hessian(:), observation_weight(:)
        type(lightgbm_options_t), intent(in) :: options
        integer, intent(in) :: rows(:)
        type(lgbm_tree_t), intent(out) :: tree
        type(fortnum_status_t), intent(out) :: status
        type(lgbm_split_t) :: candidate, best
        integer :: max_nodes, next_node, leaves, i, parent, left_node, right_node
        real(dp) :: total_g, total_h

        max_nodes = 2*options%num_leaves - 1
        allocate(tree%node(max_nodes))
        tree%n_nodes = 1
        tree%depth = 0
        allocate(tree%node(1)%rows(size(rows)))
        tree%node(1)%rows = rows
        total_g = sum(gradient(rows))
        total_h = sum(hessian(rows))
        tree%node(1)%weight = -total_g/(total_h + options%l2)
        next_node = 1
        leaves = 1
        do while (leaves < options%num_leaves)
            best%valid = .false.
            do i = 1, next_node
                if (.not. tree%node(i)%leaf .or. .not. allocated(tree%node(i)%rows)) cycle
                call best_split_for_leaf(x, gradient, hessian, observation_weight, options, &
                    tree%node(i)%rows, i, tree%node(i)%depth, candidate, status)
                if (status%code /= FORTNUM_OK) return
                if (candidate%valid) then
                    if (.not. best%valid .or. better_split(candidate, best)) then
                        call clear_split(best)
                        best = candidate
                    else
                        call clear_split(candidate)
                    end if
                end if
            end do
            if (.not. best%valid) exit
            parent = best%node
            next_node = next_node + 1
            left_node = next_node
            next_node = next_node + 1
            right_node = next_node
            tree%n_nodes = next_node
            tree%node(parent)%leaf = .false.
            tree%node(parent)%feature = best%feature
            tree%node(parent)%threshold = best%threshold
            tree%node(parent)%gain = best%gain
            tree%node(parent)%left_child = left_node
            tree%node(parent)%right_child = right_node
            deallocate(tree%node(parent)%rows)
            tree%node(left_node)%depth = tree%node(parent)%depth + 1
            tree%node(right_node)%depth = tree%node(parent)%depth + 1
            tree%depth = max(tree%depth, tree%node(left_node)%depth)
            tree%node(left_node)%weight = best%left_weight
            tree%node(right_node)%weight = best%right_weight
            allocate(tree%node(left_node)%rows(size(best%left_rows)), &
                tree%node(right_node)%rows(size(best%right_rows)))
            tree%node(left_node)%rows = best%left_rows
            tree%node(right_node)%rows = best%right_rows
            call clear_split(best)
            leaves = leaves + 1
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine build_leafwise_tree

    subroutine best_split_for_leaf(x, gradient, hessian, observation_weight, options, rows, node_id, depth, best, status)
        real(dp), intent(in) :: x(:, :), gradient(:), hessian(:), observation_weight(:)
        type(lightgbm_options_t), intent(in) :: options
        integer, intent(in) :: rows(:), node_id, depth
        type(lgbm_split_t), intent(out) :: best
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: values(:), finite_values(:)
        integer, allocatable :: order(:), finite_rows(:), positions(:)
        integer :: feature, i, n, nfinite, npositions, p, k, left_count
        real(dp) :: total_g, total_h, left_g, left_h, right_g, right_h, gain, threshold
        real(dp), allocatable :: left_rows(:)

        best%valid = .false.
        if (size(rows) < 2*options%min_data_in_leaf .or. &
            (options%max_depth > 0 .and. depth >= options%max_depth)) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if
        n = size(rows)
        total_g = sum(gradient(rows))
        total_h = sum(hessian(rows))
        allocate(values(n), finite_values(n), order(n), finite_rows(n), positions(max(1,n-1)))
        do feature = 1, size(x, 2)
            nfinite = 0
            do i = 1, n
                if (.not. ieee_is_finite(x(rows(i), feature))) cycle
                nfinite = nfinite + 1
                finite_values(nfinite) = x(rows(i), feature)
                finite_rows(nfinite) = rows(i)
            end do
            if (nfinite < 2*options%min_data_in_leaf) cycle
            call sort_indices(finite_values(:nfinite), order(:nfinite))
            call xgb_histogram_cut_positions(finite_values(:nfinite), order(:nfinite), &
                finite_rows(:nfinite), observation_weight, options%max_bin, positions, npositions)
            left_g = 0.0_dp
            left_h = 0.0_dp
            p = 1
            do k = 1, nfinite-1
                i = order(k)
                left_g = left_g + gradient(finite_rows(i))
                left_h = left_h + hessian(finite_rows(i))
                if (p > npositions) cycle
                if (positions(p) /= k) cycle
                p = p + 1
                if (finite_values(order(k)) >= finite_values(order(k+1))) cycle
                left_count = k
                if (left_count < options%min_data_in_leaf .or. &
                    nfinite-left_count < options%min_data_in_leaf) cycle
                right_g = total_g-left_g
                right_h = total_h-left_h
                if (left_h <= 0.0_dp .or. right_h <= 0.0_dp) cycle
                gain = 0.5_dp*(left_g*left_g/(left_h+options%l2) + &
                    right_g*right_g/(right_h+options%l2) - &
                    total_g*total_g/(total_h+options%l2))
                threshold = 0.5_dp*finite_values(order(k)) + &
                    0.5_dp*finite_values(order(k+1))
                if (gain <= options%min_gain_to_split) cycle
                if (.not. best%valid .or. gain > best%gain + 1.0e-14_dp .or. &
                    (abs(gain-best%gain) <= 1.0e-14_dp .and. &
                    (feature < best%feature .or. (feature == best%feature .and. threshold < best%threshold)))) then
                    call clear_split(best)
                    best%valid = .true.
                    best%feature = feature
                    best%node = node_id
                    best%threshold = threshold
                    best%gain = gain
                    best%left_weight = -left_g/(left_h+options%l2)
                    best%right_weight = -right_g/(right_h+options%l2)
                    allocate(best%left_rows(left_count), best%right_rows(nfinite-left_count))
                    best%left_rows = finite_rows(order(:left_count))
                    best%right_rows = finite_rows(order(left_count+1:nfinite))
                end if
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine best_split_for_leaf

    logical function better_split(a, b) result(value)
        type(lgbm_split_t), intent(in) :: a, b
        value = a%gain > b%gain + 1.0e-14_dp .or. &
            (abs(a%gain-b%gain) <= 1.0e-14_dp .and. &
            (a%node < b%node .or. (a%node == b%node .and. &
            (a%feature < b%feature .or. (a%feature == b%feature .and. a%threshold < b%threshold)))))
    end function better_split

    subroutine clear_split(split)
        type(lgbm_split_t), intent(inout) :: split
        if (allocated(split%left_rows)) deallocate(split%left_rows)
        if (allocated(split%right_rows)) deallocate(split%right_rows)
        split%valid = .false.
    end subroutine clear_split

    subroutine lgbm_tree_predict(tree, x, values, status)
        type(lgbm_tree_t), intent(in) :: tree
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, node

        if (tree%n_nodes < 1 .or. size(values) /= size(x, 1) .or. any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, "lightgbm tree prediction: shape or finite-input contract failed")
            return
        end if
        do i = 1, size(x, 1)
            node = 1
            do while (.not. tree%node(node)%leaf)
                if (x(i, tree%node(node)%feature) < tree%node(node)%threshold) then
                    node = tree%node(node)%left_child
                else
                    node = tree%node(node)%right_child
                end if
            end do
            values(i) = tree%node(node)%weight
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine lgbm_tree_predict

    logical function leaf_boundary_hit(self, x) result(hit)
        class(lightgbm_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer :: t, n, f
        real(dp) :: scale

        hit = .false.
        do t = 1, self%n_estimators
            do n = 1, self%estimator(t)%n_nodes
                if (self%estimator(t)%node(n)%leaf) cycle
                f = self%estimator(t)%node(n)%feature
                scale = max(1.0_dp, abs(self%estimator(t)%node(n)%threshold))
                if (any(abs(x(:, f)-self%estimator(t)%node(n)%threshold) <= 1.0e-12_dp*scale)) then
                    hit = .true.
                    return
                end if
            end do
        end do
    end function leaf_boundary_hit

    integer function parse_lgbm_objective(name) result(code)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: normalized
        normalized = trim(adjustl(name))
        select case (normalized)
        case ("regression", "reg:squarederror", "squared")
            code = LIGHTGBM_OBJECTIVE_REGRESSION
        case ("binary", "binary_logloss", "binary:logistic", "logistic")
            code = LIGHTGBM_OBJECTIVE_BINARY
        case default
            code = 0
        end select
    end function parse_lgbm_objective

    subroutine sort_indices(values, order)
        real(dp), intent(in) :: values(:)
        integer, intent(out) :: order(:)
        integer :: i, j, key
        do i = 1, size(values)
            order(i) = i
        end do
        do i = 2, size(values)
            key = order(i)
            j = i-1
            do while (j >= 1)
                if (values(order(j)) <= values(key)) exit
                order(j+1) = order(j)
                j = j-1
            end do
            order(j+1) = key
        end do
    end subroutine sort_indices

    real(dp) function stable_logit(probability) result(value)
        real(dp), intent(in) :: probability
        real(dp) :: p
        p = min(max(probability, 1.0e-12_dp), 1.0_dp-1.0e-12_dp)
        value = log(p)-log(1.0_dp-p)
    end function stable_logit

    pure real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value
        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp+exp(-value))
        else
            probability = exp(value)/(1.0_dp+exp(value))
        end if
    end function stable_sigmoid

    pure function stable_sigmoid_array(values) result(probabilities)
        real(dp), intent(in) :: values(:)
        real(dp) :: probabilities(size(values))
        integer :: i
        do i = 1, size(values)
            probabilities(i) = stable_sigmoid(values(i))
        end do
    end function stable_sigmoid_array

end module fortml_lightgbm
