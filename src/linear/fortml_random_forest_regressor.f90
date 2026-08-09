!> Deterministic bootstrap-ensemble regression built from CART trees.
module fortml_random_forest_regressor
    !! A compact, production-oriented random-forest regressor.
    !!
    !! Every tree is fit on a seeded bootstrap sample.  Targets are stored as
    !! columns, so one fitted forest serves scalar and multi-output regression
    !! without silently flattening the target state.  CART routing is
    !! piecewise constant: fixed-state input products are exact zero away from
    !! split boundaries and return a typed domain refusal on a boundary.
    !! Resident CUDA execution is an explicit typed refusal until a flattened
    !! regression-forest kernel is linked; there is no hidden host fallback.
    use, intrinsic :: iso_fortran_env, only: int64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, status_ok, &
        FORTNUM_OK, FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_tree, only: cart_regressor_t
    implicit none
    private

    integer, parameter, public :: RANDOM_FOREST_REGRESSION_MAX_TREES = 256
    integer, parameter, public :: RANDOM_FOREST_REGRESSION_DEFAULT_SEED = 5489
    integer, parameter, public :: RANDOM_FOREST_REGRESSION_MODEL_SCHEMA_VERSION = 1

    type, public :: random_forest_regressor_t
        private
        type(cart_regressor_t), allocatable :: trees(:, :)
        logical, allocatable :: bootstrap_included(:, :)
        integer :: n_inputs = 0
        integer :: n_outputs = 0
        integer :: n_samples = 0
        integer :: n_trees = 0
        integer :: max_depth = 0
        integer :: min_samples_leaf = 1
        integer :: seed = RANDOM_FOREST_REGRESSION_DEFAULT_SEED
        logical :: initialized = .false.
    contains
        procedure, public :: fit => random_forest_regressor_fit
        procedure, public :: predict => random_forest_regressor_predict
        procedure, public :: predict_staged => random_forest_regressor_predict_staged
        procedure, public :: predict_jvp => random_forest_regressor_predict_jvp
        procedure, public :: predict_vjp => random_forest_regressor_predict_vjp
        procedure, public :: predict_device => random_forest_regressor_predict_device
        procedure, public :: feature_importances => &
            random_forest_regressor_feature_importances
        procedure, public :: bootstrap_inclusion => &
            random_forest_regressor_bootstrap_inclusion
        procedure, public :: feature_count => random_forest_regressor_feature_count
        procedure, public :: output_count => random_forest_regressor_output_count
        procedure, public :: sample_count => random_forest_regressor_sample_count
        procedure, public :: tree_count => random_forest_regressor_tree_count
        procedure, public :: depth => random_forest_regressor_depth
        procedure, public :: min_leaf => random_forest_regressor_min_leaf
        procedure, public :: random_seed => random_forest_regressor_seed
        procedure, public :: schema_version => random_forest_regressor_schema_version
        procedure, public :: fitted => random_forest_regressor_fitted
        procedure, public :: device_supported => &
            random_forest_regressor_device_supported
    end type random_forest_regressor_t

    public :: random_forest_regressor_fit
    public :: random_forest_regressor_predict
    public :: random_forest_regressor_predict_staged
    public :: random_forest_regressor_predict_jvp
    public :: random_forest_regressor_predict_vjp
    public :: random_forest_regressor_predict_device

contains

    subroutine random_forest_regressor_fit(self, x, targets, status, n_trees, &
            max_depth, min_samples_leaf, seed, sample_weight)
        class(random_forest_regressor_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), targets(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_trees, max_depth, min_samples_leaf, seed
        real(dp), intent(in), optional :: sample_weight(:)
        type(random_forest_regressor_t) :: candidate
        integer :: requested_trees, requested_depth, requested_leaf, requested_seed
        integer :: n_samples, n_features, n_outputs, tree, output, row
        integer, allocatable :: bootstrap(:)
        real(dp), allocatable :: x_boot(:, :), target_boot(:), weights(:)
        integer(int64) :: rng_state

        requested_trees = 100
        if (present(n_trees)) requested_trees = n_trees
        requested_depth = 6
        if (present(max_depth)) requested_depth = max_depth
        requested_leaf = 1
        if (present(min_samples_leaf)) requested_leaf = min_samples_leaf
        requested_seed = RANDOM_FOREST_REGRESSION_DEFAULT_SEED
        if (present(seed)) requested_seed = seed
        n_samples = size(x, 1)
        n_features = size(x, 2)
        n_outputs = size(targets, 2)

        if (n_samples < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression fit: at least two samples are required")
            return
        end if
        if (n_features < 1 .or. n_outputs < 1 .or. size(targets, 1) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression fit: input and target dimensions are invalid")
            return
        end if
        if (requested_trees < 1 .or. requested_trees > &
            RANDOM_FOREST_REGRESSION_MAX_TREES .or. requested_depth < 0 .or. &
            requested_depth > 12 .or. requested_leaf < 1 .or. &
            requested_leaf > n_samples .or. requested_seed < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression fit: hyperparameters are invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. any(.not. ieee_is_finite(targets))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression fit: inputs and targets must be finite")
            return
        end if

        allocate(weights(n_samples))
        weights = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "random forest regression fit: sample weight shape is invalid")
                return
            end if
            if (any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "random forest regression fit: sample weights must be finite and positive")
                return
            end if
            weights = sample_weight
        end if

        allocate(candidate%trees(n_outputs, requested_trees), &
            candidate%bootstrap_included(n_samples, requested_trees), &
            bootstrap(n_samples), x_boot(n_samples, n_features), &
            target_boot(n_samples))
        candidate%n_inputs = n_features
        candidate%n_outputs = n_outputs
        candidate%n_samples = n_samples
        candidate%n_trees = requested_trees
        candidate%max_depth = requested_depth
        candidate%min_samples_leaf = requested_leaf
        candidate%seed = requested_seed
        candidate%initialized = .false.
        candidate%bootstrap_included = .false.
        rng_state = int(requested_seed, int64)

        do tree = 1, requested_trees
            call bootstrap_indices(n_samples, rng_state, bootstrap)
            do row = 1, n_samples
                x_boot(row, :) = x(bootstrap(row), :)
                candidate%bootstrap_included(bootstrap(row), tree) = .true.
            end do
            do output = 1, n_outputs
                do row = 1, n_samples
                    target_boot(row) = targets(bootstrap(row), output)
                end do
                call candidate%trees(output, tree)%fit(x_boot, target_boot, status, &
                    max_depth=requested_depth, min_samples_leaf=requested_leaf, &
                    sample_weight=weights(bootstrap))
                if (.not. status_ok(status)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "random forest regression fit: CART tree failed")
                    return
                end if
            end do
        end do
        candidate%initialized = .true.
        self%trees = candidate%trees
        self%bootstrap_included = candidate%bootstrap_included
        self%n_inputs = candidate%n_inputs
        self%n_outputs = candidate%n_outputs
        self%n_samples = candidate%n_samples
        self%n_trees = candidate%n_trees
        self%max_depth = candidate%max_depth
        self%min_samples_leaf = candidate%min_samples_leaf
        self%seed = candidate%seed
        self%initialized = candidate%initialized
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_forest_regressor_fit

    subroutine random_forest_regressor_predict(self, x, targets, status)
        class(random_forest_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: targets(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :), values(:)
        integer :: tree, output

        if (.not. valid_query(self, x, targets)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression predict: model, input, or shape is invalid")
            return
        end if
        allocate(candidate(size(x, 1), self%n_outputs), values(size(x, 1)))
        candidate = 0.0_dp
        do tree = 1, self%n_trees
            do output = 1, self%n_outputs
                call self%trees(output, tree)%predict(x, values, status)
                if (.not. status_ok(status)) return
                candidate(:, output) = candidate(:, output) + values
            end do
        end do
        candidate = candidate/real(self%n_trees, dp)
        targets = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_forest_regressor_predict

    subroutine random_forest_regressor_predict_staged(self, x, staged, status)
        class(random_forest_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: staged(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :, :), values(:)
        integer :: tree, output

        if (.not. self%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression staged prediction: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(staged) /= [size(x, 1), self%n_trees, self%n_outputs])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression staged prediction: input or shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression staged prediction: inputs must be finite")
            return
        end if
        allocate(candidate(size(staged, 1), size(staged, 2), size(staged, 3)), &
            values(size(x, 1)))
        candidate = 0.0_dp
        do tree = 1, self%n_trees
            do output = 1, self%n_outputs
                call self%trees(output, tree)%predict(x, values, status)
                if (.not. status_ok(status)) return
                candidate(:, tree, output) = values
            end do
            if (tree > 1) then
                candidate(:, tree, :) = &
                    (candidate(:, tree - 1, :)*real(tree - 1, dp) + &
                    candidate(:, tree, :))/real(tree, dp)
            else
                candidate(:, tree, :) = candidate(:, tree, :)/real(tree, dp)
            end if
        end do
        staged = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_forest_regressor_predict_staged

    subroutine random_forest_regressor_predict_jvp(self, x, x_dot, targets, &
            targets_dot, status)
        class(random_forest_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(inout) :: targets(:, :), targets_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:, :), candidate_dot(:, :), values(:), &
            values_dot(:)
        integer :: tree, output

        if (.not. valid_query(self, x, targets)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression JVP: model, input, or shape is invalid")
            return
        end if
        if (any(shape(x_dot) /= shape(x)) .or. any(shape(targets_dot) /= &
            shape(targets)) .or. any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression JVP: tangent or output shape is invalid")
            return
        end if
        allocate(candidate(size(targets, 1), size(targets, 2)), &
            candidate_dot(size(targets_dot, 1), size(targets_dot, 2)), &
            values(size(x, 1)), values_dot(size(x, 1)))
        candidate = 0.0_dp
        candidate_dot = 0.0_dp
        do tree = 1, self%n_trees
            do output = 1, self%n_outputs
                call self%trees(output, tree)%predict_jvp(x, x_dot, values, &
                    values_dot, status)
                if (.not. status_ok(status)) return
                candidate(:, output) = candidate(:, output) + values
                candidate_dot(:, output) = candidate_dot(:, output) + values_dot
            end do
        end do
        candidate = candidate/real(self%n_trees, dp)
        candidate_dot = candidate_dot/real(self%n_trees, dp)
        targets = candidate
        targets_dot = candidate_dot
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_forest_regressor_predict_jvp

    subroutine random_forest_regressor_predict_vjp(self, x, targets_bar, x_bar, status)
        class(random_forest_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), targets_bar(:, :)
        real(dp), intent(inout) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:,:), x_dot(:,:), values(:), values_dot(:)
        integer :: tree, output

        if (.not. self%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression VJP: model is not fitted")
            return
        end if
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(targets_bar) /= [size(x, 1), self%n_outputs]) .or. &
            any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression VJP: input or shape is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x)) .or. &
            any(.not. ieee_is_finite(targets_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression VJP: inputs must be finite")
            return
        end if
        allocate(candidate(size(x, 1), size(x, 2)), x_dot(size(x, 1), size(x, 2)), &
            values(size(x, 1)), values_dot(size(x, 1)))
        x_dot = 0.0_dp
        candidate = 0.0_dp
        do tree = 1, self%n_trees
            do output = 1, self%n_outputs
                call self%trees(output, tree)%predict_jvp(x, x_dot, values, &
                    values_dot, status)
                if (.not. status_ok(status)) return
            end do
        end do
        x_bar = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_forest_regressor_predict_vjp

    subroutine random_forest_regressor_predict_device(self, device, x, targets, status)
        class(random_forest_regressor_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(inout) :: targets(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, targets, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "random forest regression device prediction: no resident CUDA forest kernel")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression device prediction: device kind is invalid")
        end select
    end subroutine random_forest_regressor_predict_device

    subroutine random_forest_regressor_feature_importances(self, importance, status)
        class(random_forest_regressor_t), intent(in) :: self
        real(dp), intent(inout) :: importance(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: candidate(:), tree_importance(:)
        integer :: tree, output
        real(dp) :: total

        if (.not. self%initialized .or. size(importance) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "random forest regression feature_importances: model or shape is invalid")
            return
        end if
        allocate(candidate(self%n_inputs), tree_importance(self%n_inputs))
        candidate = 0.0_dp
        do tree = 1, self%n_trees
            do output = 1, self%n_outputs
                tree_importance = 0.0_dp
                call self%trees(output, tree)%feature_importances(tree_importance, status)
                if (.not. status_ok(status)) return
                candidate = candidate + tree_importance
            end do
        end do
        total = sum(candidate)
        if (total > 0.0_dp) candidate = candidate/total
        importance = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine random_forest_regressor_feature_importances

    function random_forest_regressor_bootstrap_inclusion(self) result(inclusion)
        class(random_forest_regressor_t), intent(in) :: self
        logical, allocatable :: inclusion(:, :)
        if (allocated(self%bootstrap_included)) then
            allocate(inclusion(size(self%bootstrap_included, 1), &
                size(self%bootstrap_included, 2)))
            inclusion = self%bootstrap_included
        else
            allocate(inclusion(0, 0))
        end if
    end function random_forest_regressor_bootstrap_inclusion

    integer function random_forest_regressor_feature_count(self) result(count)
        class(random_forest_regressor_t), intent(in) :: self
        count = self%n_inputs
    end function random_forest_regressor_feature_count

    integer function random_forest_regressor_output_count(self) result(count)
        class(random_forest_regressor_t), intent(in) :: self
        count = self%n_outputs
    end function random_forest_regressor_output_count

    integer function random_forest_regressor_sample_count(self) result(count)
        class(random_forest_regressor_t), intent(in) :: self
        count = self%n_samples
    end function random_forest_regressor_sample_count

    integer function random_forest_regressor_tree_count(self) result(count)
        class(random_forest_regressor_t), intent(in) :: self
        count = self%n_trees
    end function random_forest_regressor_tree_count

    integer function random_forest_regressor_depth(self) result(depth)
        class(random_forest_regressor_t), intent(in) :: self
        depth = self%max_depth
    end function random_forest_regressor_depth

    integer function random_forest_regressor_min_leaf(self) result(value)
        class(random_forest_regressor_t), intent(in) :: self
        value = self%min_samples_leaf
    end function random_forest_regressor_min_leaf

    integer function random_forest_regressor_seed(self) result(value)
        class(random_forest_regressor_t), intent(in) :: self
        value = self%seed
    end function random_forest_regressor_seed

    integer function random_forest_regressor_schema_version(self) result(value)
        class(random_forest_regressor_t), intent(in) :: self
        value = RANDOM_FOREST_REGRESSION_MODEL_SCHEMA_VERSION
    end function random_forest_regressor_schema_version

    logical function random_forest_regressor_fitted(self) result(value)
        class(random_forest_regressor_t), intent(in) :: self
        value = self%initialized .and. allocated(self%trees) .and. &
            self%n_trees > 0 .and. self%n_outputs > 0
    end function random_forest_regressor_fitted

    logical function random_forest_regressor_device_supported(self, device_kind) &
            result(value)
        class(random_forest_regressor_t), intent(in) :: self
        integer, intent(in) :: device_kind
        value = device_kind == FORTML_DEVICE_CPU .and. self%fitted()
    end function random_forest_regressor_device_supported

    logical function valid_query(self, x, targets) result(valid)
        class(random_forest_regressor_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), targets(:, :)
        valid = .false.
        if (.not. self%initialized) return
        if (size(x, 1) < 1 .or. size(x, 2) /= self%n_inputs) return
        if (any(shape(targets) /= [size(x, 1), self%n_outputs])) return
        if (any(.not. ieee_is_finite(x))) return
        valid = .true.
    end function valid_query

    subroutine bootstrap_indices(n_samples, state, indices)
        integer, intent(in) :: n_samples
        integer(int64), intent(inout) :: state
        integer, intent(out) :: indices(:)
        integer :: row
        do row = 1, n_samples
            indices(row) = 1 + int(real(n_samples, dp)*forest_uniform(state))
            if (indices(row) > n_samples) indices(row) = n_samples
        end do
    end subroutine bootstrap_indices

    real(dp) function forest_uniform(state) result(value)
        integer(int64), intent(inout) :: state
        state = modulo(48271_int64*state, 2147483647_int64)
        value = real(state, dp)/2147483647.0_dp
    end function forest_uniform

end module fortml_random_forest_regressor
