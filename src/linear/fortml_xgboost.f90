!> A deterministic, exact-split second-order boosting foundation.
module fortml_xgboost
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan
    use, intrinsic :: iso_fortran_env, only: int64, iostat_end
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    implicit none
    private

    integer, parameter, public :: XGB_OBJECTIVE_SQUARED = 1
    integer, parameter, public :: XGB_OBJECTIVE_LOGISTIC = 2
    integer, parameter, public :: XGB_OBJECTIVE_POISSON = 3
    integer, parameter, public :: XGB_OBJECTIVE_HUBER = 4
    integer, parameter, public :: XGB_OBJECTIVE_QUANTILE = 5
    integer, parameter, public :: XGB_OBJECTIVE_SQUARED_LOG = 6
    integer, parameter, public :: XGB_OBJECTIVE_RANK_PAIRWISE = 7
    integer, parameter, public :: XGB_OBJECTIVE_ABSOLUTE = 8
    integer, parameter, public :: XGB_OBJECTIVE_TWEEDIE = 9
    integer, parameter, public :: XGB_MISSING_ERROR = 0
    integer, parameter, public :: XGB_MISSING_LEARN = 1
    integer, parameter, public :: XGB_MISSING_LEFT = 2
    integer, parameter, public :: XGB_MISSING_RIGHT = 3
    integer, parameter, public :: XGB_TREE_EXACT = 1
    integer, parameter, public :: XGB_TREE_HIST = 2
    integer, parameter, public :: XGB_CATEGORICAL_NONE = 0
    integer, parameter, public :: XGB_CATEGORICAL_ORDERED = 1
    character(*), parameter, public :: XGB_MODEL_TEXT_MAGIC = &
        "FORTML_XGBOOST_TEXT"
    integer, parameter, public :: XGB_MODEL_TEXT_SCHEMA_VERSION = 4
    integer, parameter :: XGB_MAX_SERIALIZED_NODES = 1000000
    integer, parameter :: XGB_MAX_CATEGORICAL_VALUES = 64

    public :: xgb_pairwise_loss, xgb_pairwise_derivatives
    public :: xgb_histogram_cut_positions
    public :: xgb_tweedie_loss, xgb_tweedie_derivatives

    !> Options for the deterministic exact- or histogram-split
    !> XGBoost-style estimator.
    !>
    !> Numeric splits are enumerated exhaustively.  Trees may grow to
    !> `max_depth` and use the full second-order leaf and split formulas,
    !> including L1/L2 regularisation, gamma, min-child-Hessian and shrinkage.
    !> The exact CPU path accepts IEEE NaNs when `missing_policy` is `learn`,
    !> `left`, or `right`.  `learn` evaluates both default directions for every
    !> candidate split and stores the direction with the best gain, while the
    !> fixed policies route all missing values to the requested child.
    !>
    !> `monotone_constraints`, when allocated, contains one entry per input
    !> feature and uses the XGBoost convention `-1`, `0`, and `+1` for
    !> decreasing, unconstrained, and increasing predictions.  Constraints
    !> are enforced per tree by propagating leaf-value bounds through every
    !> recursive split.  Fit remains piecewise/discrete; input products keep
    !> the existing split-boundary refusal contract.
    !>
    !> `interaction_groups`, when allocated, assigns each feature to an
    !> interaction group. A zero entry leaves that feature unconstrained;
    !> after a positive-group feature is used on a root-to-leaf path, all
    !> descendant splits on that path must use features from the same group.
    type, public :: xgboost_options_t
        integer :: n_estimators = 50
        integer :: max_depth = 1
        integer :: min_samples_leaf = 1
        real(dp) :: learning_rate = 0.3_dp
        real(dp) :: l1 = 0.0_dp
        real(dp) :: l2 = 1.0_dp
        real(dp) :: gamma = 0.0_dp
        real(dp) :: min_child_weight = 1.0e-3_dp
        ! XGBoost's canonical names include `reg:squaredlogerror` and
        ! `reg:pseudohubererror`; keep enough storage for the full names so
        ! aliases are never silently truncated before parsing.
        character(len=32) :: objective = "squared"
        real(dp) :: huber_delta = 1.0_dp
        real(dp) :: quantile_alpha = 0.5_dp
        !! Tweedie variance power.  The bounded production path supports
        !! 1 < power < 2, matching XGBoost's compound-Poisson regime.
        real(dp) :: tweedie_variance_power = 1.5_dp
        character(len=16) :: missing_policy = "error"
        character(len=16) :: tree_method = "exact"
        integer :: max_bin = 256
        !! Integer-coded categorical feature indices (one-based, sorted).
        !! The bounded ordered-gradient policy is selected with
        !! `categorical_policy="ordered"`; categories beyond
        !! `categorical_max_categories` are refused explicitly.
        character(len=16) :: categorical_policy = "none"
        integer :: categorical_max_categories = 8
        integer, allocatable :: categorical_features(:)
        !! Evaluate an optional validation set after every boosting round.
        !! A positive value stops after this many consecutive rounds without
        !! an improvement larger than `early_stopping_min_delta`.
        integer :: early_stopping_rounds = 0
        real(dp) :: early_stopping_min_delta = 0.0_dp
        logical :: restore_best = .true.
        !! Fraction of training rows retained independently for each tree.
        !! Sampling is without replacement and uses the local `seed` stream.
        real(dp) :: subsample = 1.0_dp
        !! Fraction of input features considered independently for each tree.
        !! Selected feature indices are traversed in ascending order.
        real(dp) :: colsample_bytree = 1.0_dp
        !! Positive local stream seed for deterministic row/feature sampling.
        integer(int64) :: seed = 104729_int64
        integer, allocatable :: monotone_constraints(:)
        integer, allocatable :: interaction_groups(:)
    end type xgboost_options_t

    type :: xgb_tree_t
        ! Legacy root fields are retained as cheap diagnostics for the public
        ! split-gain/leaf-weight accessors.  The complete tree is represented
        ! by the node arrays below.
        integer :: feature_index = 0
        integer :: left_count = 0
        integer :: right_count = 0
        real(dp) :: threshold = 0.0_dp
        real(dp) :: left_weight = 0.0_dp
        real(dp) :: right_weight = 0.0_dp
        real(dp) :: split_gain = 0.0_dp
        logical :: has_split = .false.
        integer :: n_nodes = 0
        integer :: depth = 0
        integer, allocatable :: feature(:), left_child(:), right_child(:)
        real(dp), allocatable :: node_threshold(:), weight(:), node_gain(:)
        real(dp), allocatable :: node_cover(:)
        logical, allocatable :: leaf(:)
        logical, allocatable :: missing_left(:)
        logical, allocatable :: categorical(:)
        integer, allocatable :: category_count(:), category_values(:, :)
    end type xgb_tree_t

    !> Second-order boosting for squared, squared-log (RMSLE), binary
    !> logistic, Poisson count, Tweedie, Huber, quantile, and absolute objectives.
    !> Fit is discrete; predictions are deterministic and the objective's
    !> Hessians are aggregated exactly for every retained candidate split.
    type, public :: xgboost_t
        private
        integer :: n_inputs = 0
        integer :: n_estimators = 0
        integer :: requested_estimators = 0
        integer :: objective_code = 0
        integer :: tree_method_code = XGB_TREE_EXACT
        integer :: max_bin = 256
        integer :: max_depth_value = 0
        integer :: min_samples_leaf_value = 0
        integer :: early_stopping_rounds_value = 0
        real(dp) :: learning_rate = 0.0_dp
        real(dp) :: l1_value = 0.0_dp
        real(dp) :: l2_value = 0.0_dp
        real(dp) :: gamma_value = 0.0_dp
        real(dp) :: min_child_weight_value = 0.0_dp
        real(dp) :: early_stopping_min_delta_value = 0.0_dp
        real(dp) :: subsample_value = 1.0_dp
        real(dp) :: colsample_bytree_value = 1.0_dp
        integer(int64) :: seed_value = 0_int64
        logical :: restore_best_value = .true.
        real(dp) :: base_score = 0.0_dp
        real(dp) :: objective_parameter = 0.0_dp
        integer :: best_iteration_value = 0
        real(dp) :: best_validation_loss_value = huge(1.0_dp)
        logical :: early_stopped_flag = .false.
        integer :: missing_code = XGB_MISSING_ERROR
        integer :: categorical_policy_code = XGB_CATEGORICAL_NONE
        integer :: categorical_max_categories_value = 0
        integer, allocatable :: categorical_features(:)
        integer, allocatable :: monotone_constraints(:)
        integer, allocatable :: interaction_groups(:)
        type(xgb_tree_t), allocatable :: estimators(:)
        logical :: initialized = .false.
    contains
        procedure, public :: fit => xgb_fit
        procedure, public :: fit_warm_start => xgb_fit_warm_start
        procedure, public :: fit_regression => xgb_fit_regression
        procedure, public :: fit_binary => xgb_fit_binary
        procedure, public :: fit_poisson => xgb_fit_poisson
        procedure, public :: fit_tweedie => xgb_fit_tweedie
        procedure, public :: fit_huber => xgb_fit_huber
        procedure, public :: fit_quantile => xgb_fit_quantile
        procedure, public :: fit_squared_log => xgb_fit_squared_log
        procedure, public :: fit_absolute => xgb_fit_absolute
        procedure, public :: fit_ranking => xgb_fit_ranking
        procedure, public :: predict_matrix => xgb_predict_matrix
        procedure, public :: predict_vector => xgb_predict_vector
        generic, public :: predict => predict_matrix, predict_vector
        procedure, public :: predict_device_vector => xgb_predict_device
        procedure, public :: predict_device_matrix => xgb_predict_device_matrix
        generic, public :: predict_device => predict_device_vector, &
            predict_device_matrix
        procedure, public :: device_supported => xgb_device_supported
        procedure, public :: predict_margin_matrix => xgb_predict_margin_matrix
        procedure, public :: predict_margin_vector => xgb_predict_margin_vector
        generic, public :: predict_margin => predict_margin_matrix, &
            predict_margin_vector
        procedure, public :: predict_jvp => xgb_predict_jvp
        procedure, public :: predict_staged => xgb_predict_staged
        procedure, public :: predict_staged_margin => xgb_predict_staged_margin
        procedure, public :: slice => xgb_slice
        procedure, public :: predict_contributions => xgb_predict_contributions
        procedure, public :: predict_contributions_device => &
            xgb_predict_contributions_device
        procedure, public :: predict_proba_staged => xgb_predict_proba_staged
        procedure, public :: predict_proba => xgb_predict_proba
        procedure, public :: decision_function => xgb_decision_function
        procedure, public :: predict_vjp => xgb_predict_vjp
        procedure, public :: split_gain => xgb_split_gain
        procedure, public :: leaf_weights => xgb_leaf_weights
        procedure, public :: feature_importance => xgb_feature_importance
        procedure, public :: tree_node_count => xgb_tree_node_count
        procedure, public :: tree_depth => xgb_tree_depth
        procedure, public :: feature_count => xgb_feature_count
        procedure, public :: estimator_count => xgb_estimator_count
        procedure, public :: requested_estimator_count => xgb_requested_estimator_count
        procedure, public :: base_margin => xgb_base_margin
        procedure, public :: objective_name => xgb_objective_name
        procedure, public :: objective_parameter_value => xgb_objective_parameter
        procedure, public :: missing_policy => xgb_missing_policy
        procedure, public :: tree_method => xgb_tree_method
        procedure, public :: max_bin_count => xgb_max_bin_count
        procedure, public :: accepts_missing => xgb_accepts_missing
        procedure, public :: categorical_policy => xgb_categorical_policy
        procedure, public :: categorical_max_categories => xgb_categorical_max_categories
        procedure, public :: categorical_feature => xgb_categorical_feature
        procedure, public :: monotone_constraint => xgb_monotone_constraint
        procedure, public :: interaction_group => xgb_interaction_group
        procedure, public :: fitted => xgb_fitted
        procedure, public :: best_iteration => xgb_best_iteration
        procedure, public :: best_validation_loss => xgb_best_validation_loss
        procedure, public :: early_stopped => xgb_early_stopped
        procedure, public :: save_text => xgb_save_text
        procedure, public :: load_text => xgb_load_text
    end type xgboost_t

contains

    !> Fit the `rank:pairwise` objective using integer query/group IDs.
    !! Rows sharing one ID form one ranking query; pairs from different
    !! queries never contribute to the gradient, Hessian, or validation loss.
    !! The fitted model predicts raw ranking margins (there is no probability
    !! link).  Optional sample weights use the smaller weight of each pair.
    subroutine xgb_fit_ranking(self, x, y, group, status, options, sample_weight, &
            validation_x, validation_y, validation_group, validation_weight)
        class(xgboost_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        integer, intent(in) :: group(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_y(:)
        integer, intent(in), optional :: validation_group(:)
        real(dp), intent(in), optional :: validation_weight(:)
        type(xgboost_options_t) :: settings

        settings = xgboost_options_t()
        if (present(options)) settings = options
        settings%objective = "rank:pairwise"
        call xgb_fit(self, x, y, status, settings, sample_weight, &
            validation_x, validation_y, validation_weight, group, validation_group)
    end subroutine xgb_fit_ranking

    subroutine xgb_fit(self, x, y, status, options, sample_weight, &
            validation_x, validation_y, validation_weight, group, validation_group)
        class(xgboost_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_y(:), &
            validation_weight(:)
        integer, intent(in), optional :: group(:), validation_group(:)
        type(xgboost_options_t) :: settings
        real(dp), allocatable :: prediction(:), gradient(:), hessian(:)
        real(dp), allocatable :: correction(:)
        real(dp), allocatable :: observation_weight(:)
        real(dp), allocatable :: validation_prediction(:), validation_correction(:)
        real(dp), allocatable :: validation_observation_weight(:)
        type(xgb_tree_t), allocatable :: best_estimators(:), retained_estimators(:)
        real(dp) :: mean_target, rate, weight_sum
        real(dp) :: validation_loss, best_validation_loss
        integer :: objective_code, missing_code, tree_method_code, categorical_policy_code
        integer :: i, n_samples
        integer :: n_features, n_validation, completed_estimators
        integer :: best_iteration, stale_rounds
        integer(int64) :: sampling_state
        integer, allocatable :: sample_index(:)
        logical, allocatable :: feature_mask(:)
        logical :: have_validation, improved, is_ranking

        settings = xgboost_options_t()
        if (present(options)) settings = options
        have_validation = present(validation_x) .or. present(validation_y) .or. &
            present(validation_weight)
        if (present(validation_x) .neqv. present(validation_y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: validation_x and validation_y must be supplied together")
            return
        end if
        if (present(validation_weight) .and. .not. present(validation_x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: validation_weight requires validation data")
            return
        end if
        if (settings%early_stopping_rounds < 0 .or. &
            .not. ieee_is_finite(settings%early_stopping_min_delta) .or. &
            settings%early_stopping_min_delta < 0.0_dp .or. &
            (settings%early_stopping_rounds > 0 .and. .not. have_validation)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: invalid early-stopping configuration")
            return
        end if
        objective_code = parse_objective(settings%objective)
        if (objective_code == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: unsupported objective")
            return
        end if
        is_ranking = objective_code == XGB_OBJECTIVE_RANK_PAIRWISE
        if (is_ranking .neqv. present(group)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: rank:pairwise requires group IDs and other objectives reject them")
            return
        end if
        if (present(validation_group) .and. .not. is_ranking) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: validation group IDs are only valid for rank:pairwise")
            return
        end if
        if (is_ranking .and. have_validation .and. .not. present(validation_group)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: ranking validation data requires validation group IDs")
            return
        end if
        if (is_ranking .and. .not. valid_group_ids(group, size(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: group IDs must be positive and match the target length")
            return
        end if
        missing_code = parse_missing_policy(settings%missing_policy)
        if (missing_code < XGB_MISSING_ERROR) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: missing_policy must be error, learn, left, or right")
            return
        end if
        tree_method_code = parse_tree_method(settings%tree_method)
        if (tree_method_code == 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: tree_method must be exact or hist")
            return
        end if
        categorical_policy_code = parse_categorical_policy(settings%categorical_policy)
        if (categorical_policy_code < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: categorical_policy must be none or ordered")
            return
        end if
        if (categorical_policy_code == 0 .and. allocated(settings%categorical_features)) then
            if (size(settings%categorical_features) > 0) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: categorical_features require categorical_policy=ordered")
                return
            end if
        end if

        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (have_validation) then
            n_validation = size(validation_x, 1)
            if (n_validation < 1 .or. size(validation_x, 2) /= n_features .or. &
                size(validation_y) /= n_validation) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: validation dimensions do not match training features")
                return
            end if
            if (is_ranking .and. .not. valid_group_ids(validation_group, n_validation)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: validation group IDs must be positive and match validation targets")
                return
            end if
        else
            n_validation = 0
        end if
        rate = settings%learning_rate
        if (n_samples < 2 .or. n_features < 1 .or. size(y) /= n_samples .or. &
            settings%n_estimators < 1 .or. settings%max_depth < 1 .or. &
            settings%max_depth > n_samples .or. &
            settings%min_samples_leaf < 1 .or. &
            2*settings%min_samples_leaf > n_samples .or. &
            .not. ieee_is_finite(rate) .or. rate <= 0.0_dp .or. rate > 1.0_dp .or. &
            .not. ieee_is_finite(settings%l1) .or. settings%l1 < 0.0_dp .or. &
            .not. ieee_is_finite(settings%l2) .or. settings%l2 < 0.0_dp .or. &
            .not. ieee_is_finite(settings%gamma) .or. settings%gamma < 0.0_dp .or. &
            .not. ieee_is_finite(settings%min_child_weight) .or. &
            settings%min_child_weight < 0.0_dp .or. &
            .not. ieee_is_finite(settings%subsample) .or. settings%subsample <= 0.0_dp .or. &
            settings%subsample > 1.0_dp .or. &
            .not. ieee_is_finite(settings%colsample_bytree) .or. &
            settings%colsample_bytree <= 0.0_dp .or. &
            settings%colsample_bytree > 1.0_dp .or. settings%seed <= 0_int64 .or. &
            (tree_method_code == XGB_TREE_HIST .and. settings%max_bin < 2) .or. &
            (categorical_policy_code == XGB_CATEGORICAL_ORDERED .and. &
             (settings%categorical_max_categories < 2 .or. &
              settings%categorical_max_categories > XGB_MAX_CATEGORICAL_VALUES))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: invalid dimensions or hyperparameters")
            return
        end if
        if (categorical_policy_code == XGB_CATEGORICAL_ORDERED) then
            if (.not. allocated(settings%categorical_features) .or. &
                size(settings%categorical_features) < 1 .or. &
                any(settings%categorical_features < 1) .or. &
                any(settings%categorical_features > n_features) .or. &
                has_duplicate_sorted_index(settings%categorical_features)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: categorical_features must be unique one-based feature indices")
                return
            end if
        end if
        if (allocated(settings%monotone_constraints)) then
            if (size(settings%monotone_constraints) /= n_features .or. &
                any(abs(settings%monotone_constraints) > 1)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: monotone_constraints must have one entry per feature in {-1,0,1}")
                return
            end if
        end if
        if (allocated(settings%interaction_groups)) then
            if (size(settings%interaction_groups) /= n_features .or. &
                any(settings%interaction_groups < 0)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: interaction_groups must have one nonnegative entry per feature")
                return
            end if
        end if
        if (any((.not. ieee_is_finite(x)) .and. (.not. ieee_is_nan(x))) .or. &
            any(.not. ieee_is_finite(y))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: inputs must be finite or IEEE NaN and targets finite")
            return
        end if
        if (categorical_policy_code == XGB_CATEGORICAL_ORDERED) then
            if (.not. valid_categorical_values(x, settings%categorical_features, &
                    settings%categorical_max_categories)) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "xgboost fit: categorical values must be finite integer codes within max category bound")
                return
            end if
            if (have_validation) then
                if (.not. valid_categorical_values(validation_x, settings%categorical_features, &
                        settings%categorical_max_categories)) then
                    call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                        "xgboost fit: validation categorical values exceed integer/max-category boundary")
                    return
                end if
            end if
        end if
        if (have_validation) then
            if (any((.not. ieee_is_finite(validation_x)) .and. &
                (.not. ieee_is_nan(validation_x))) .or. &
                any(.not. ieee_is_finite(validation_y))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: validation inputs must be finite or IEEE NaN and targets finite")
                return
            end if
        end if
        if (missing_code == XGB_MISSING_ERROR .and. any(ieee_is_nan(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: NaN inputs require a missing-value policy")
            return
        end if
        if (have_validation) then
            if (missing_code == XGB_MISSING_ERROR .and. any(ieee_is_nan(validation_x))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: validation NaN inputs require a missing-value policy")
                return
            end if
        end if
        if (objective_code == XGB_OBJECTIVE_LOGISTIC) then
            if (any(y < 0.0_dp) .or. any(y > 1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: logistic targets must be in [0, 1]")
                return
            end if
            if (have_validation) then
                if (any(validation_y < 0.0_dp) .or. any(validation_y > 1.0_dp)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "xgboost fit: validation logistic targets must be in [0, 1]")
                    return
                end if
            end if
        else if (objective_code == XGB_OBJECTIVE_POISSON .or. &
            objective_code == XGB_OBJECTIVE_TWEEDIE) then
            if (any(y < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: count/Tweedie targets must be nonnegative")
                return
            end if
            if (have_validation) then
                if (any(validation_y < 0.0_dp)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "xgboost fit: validation count/Tweedie targets must be nonnegative")
                    return
                end if
            end if
        else if (objective_code == XGB_OBJECTIVE_SQUARED_LOG) then
            if (any(y < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: squared-log targets must be nonnegative")
                return
            end if
            if (have_validation) then
                if (any(validation_y < 0.0_dp)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "xgboost fit: validation squared-log targets must be nonnegative")
                    return
                end if
            end if
        end if
        if (objective_code == XGB_OBJECTIVE_HUBER) then
            if (.not. ieee_is_finite(settings%huber_delta) .or. &
                settings%huber_delta <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: huber_delta must be finite and positive")
                return
            end if
        else if (objective_code == XGB_OBJECTIVE_QUANTILE) then
            if (.not. ieee_is_finite(settings%quantile_alpha) .or. &
                settings%quantile_alpha <= 0.0_dp .or. &
                settings%quantile_alpha >= 1.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: quantile_alpha must lie strictly between zero and one")
                return
            end if
        else if (objective_code == XGB_OBJECTIVE_TWEEDIE) then
            if (.not. ieee_is_finite(settings%tweedie_variance_power) .or. &
                settings%tweedie_variance_power <= 1.0_dp .or. &
                settings%tweedie_variance_power >= 2.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: tweedie_variance_power must lie strictly between one and two")
                return
            end if
        end if

        allocate(observation_weight(n_samples))
        observation_weight = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: sample_weight must be positive and finite")
                return
            end if
            observation_weight = sample_weight
        end if
        weight_sum = sum(observation_weight)
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: sample_weight has no positive mass")
            return
        end if
        if (have_validation) then
            allocate(validation_observation_weight(n_validation))
            validation_observation_weight = 1.0_dp
            if (present(validation_weight)) then
                if (size(validation_weight) /= n_validation .or. &
                    any(.not. ieee_is_finite(validation_weight)) .or. &
                    any(validation_weight <= 0.0_dp)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "xgboost fit: validation_weight must be positive and finite")
                    return
                end if
                validation_observation_weight = validation_weight
            end if
            if (.not. ieee_is_finite(sum(validation_observation_weight)) .or. &
                sum(validation_observation_weight) <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: validation_weight has no positive mass")
                return
            end if
        end if

        mean_target = sum(observation_weight*y)/weight_sum
        if (.not. ieee_is_finite(mean_target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost fit: weighted target mean is nonfinite")
            return
        end if
        if (objective_code == XGB_OBJECTIVE_RANK_PAIRWISE) then
            ! Pairwise ranking is invariant to an additive margin shift.  A
            ! zero link intercept keeps the initial pair margins exactly zero.
            self%base_score = 0.0_dp
        else if (objective_code == XGB_OBJECTIVE_LOGISTIC) then
            self%base_score = stable_logit(mean_target)
        else if (objective_code == XGB_OBJECTIVE_POISSON) then
            self%base_score = stable_log_rate(mean_target)
        else if (objective_code == XGB_OBJECTIVE_SQUARED_LOG) then
            ! The constant optimum is the weighted geometric mean in the
            ! transformed `log(1+y)` coordinate, not `log(1+mean(y))`.
            self%base_score = sum(observation_weight*log(1.0_dp + y))/weight_sum
            if (.not. ieee_is_finite(self%base_score)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: squared-log base margin is nonfinite")
                return
            end if
        else if (objective_code == XGB_OBJECTIVE_QUANTILE) then
            self%base_score = weighted_quantile(y, observation_weight, &
                settings%quantile_alpha)
        else if (objective_code == XGB_OBJECTIVE_ABSOLUTE) then
            self%base_score = weighted_quantile(y, observation_weight, 0.5_dp)
        else
            self%base_score = mean_target
        end if
        allocate(self%estimators(settings%n_estimators))
        allocate(prediction(n_samples), gradient(n_samples), hessian(n_samples))
        allocate(correction(n_samples))
        prediction = self%base_score
        if (have_validation) then
            allocate(validation_prediction(n_validation), validation_correction(n_validation))
            validation_prediction = self%base_score
            best_validation_loss = huge(1.0_dp)
            best_iteration = 0
            stale_rounds = 0
        else
            best_validation_loss = huge(1.0_dp)
            best_iteration = settings%n_estimators
            stale_rounds = 0
        end if

        completed_estimators = 0
        sampling_state = settings%seed
        do i = 1, settings%n_estimators
            call objective_derivatives(objective_code, prediction, y, gradient, &
                hessian, settings%huber_delta, settings%quantile_alpha, &
                settings%tweedie_variance_power, status, &
                group, observation_weight)
            if (status%code /= FORTNUM_OK) return
            if (is_ranking) hessian = max(hessian, 1.0e-12_dp)
            if (.not. is_ranking) then
                gradient = observation_weight*gradient
                hessian = observation_weight*hessian
            end if
            call sample_training_rows(n_samples, settings%subsample, sampling_state, &
                sample_index)
            call sample_training_features(n_features, settings%colsample_bytree, &
                sampling_state, feature_mask)
            call build_tree(x, gradient, hessian, observation_weight, settings, &
                sample_index, feature_mask, self%estimators(i), status)
            if (status%code /= FORTNUM_OK) return
            call tree_predict(self%estimators(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            prediction = prediction + rate*correction
            completed_estimators = i
            if (have_validation) then
                call tree_predict(self%estimators(i), validation_x, &
                    validation_correction, status)
                if (status%code /= FORTNUM_OK) return
                validation_prediction = validation_prediction + rate*validation_correction
                call xgb_objective_loss(objective_code, validation_prediction, &
                    validation_y, validation_observation_weight, settings%huber_delta, &
                    settings%quantile_alpha, settings%tweedie_variance_power, &
                    validation_loss, status, validation_group)
                if (status%code /= FORTNUM_OK) return
                improved = validation_loss < best_validation_loss - &
                    settings%early_stopping_min_delta
                if (improved) then
                    best_validation_loss = validation_loss
                    best_iteration = i
                    stale_rounds = 0
                    if (settings%restore_best) best_estimators = self%estimators(:i)
                else
                    stale_rounds = stale_rounds + 1
                end if
                if (settings%early_stopping_rounds > 0 .and. &
                    stale_rounds >= settings%early_stopping_rounds) then
                    self%early_stopped_flag = .true.
                    exit
                end if
            end if
        end do

        if (have_validation) then
            if (best_iteration < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost fit: validation objective did not produce a finite score")
                return
            end if
            self%best_iteration_value = best_iteration
            self%best_validation_loss_value = best_validation_loss
            if (settings%restore_best .and. allocated(best_estimators)) then
                call retain_xgb_estimators(self, best_estimators)
                completed_estimators = best_iteration
            else if (completed_estimators < settings%n_estimators) then
                allocate(retained_estimators(completed_estimators))
                retained_estimators = self%estimators(:completed_estimators)
                call move_alloc(retained_estimators, self%estimators)
            end if
        else
            self%best_iteration_value = settings%n_estimators
            self%best_validation_loss_value = huge(1.0_dp)
        end if

        self%n_inputs = n_features
        self%n_estimators = settings%n_estimators
        if (have_validation) self%n_estimators = completed_estimators
        self%requested_estimators = settings%n_estimators
        self%objective_code = objective_code
        self%tree_method_code = tree_method_code
        self%max_bin = settings%max_bin
        self%max_depth_value = settings%max_depth
        self%min_samples_leaf_value = settings%min_samples_leaf
        self%early_stopping_rounds_value = settings%early_stopping_rounds
        self%learning_rate = rate
        self%l1_value = settings%l1
        self%l2_value = settings%l2
        self%gamma_value = settings%gamma
        self%min_child_weight_value = settings%min_child_weight
        self%early_stopping_min_delta_value = settings%early_stopping_min_delta
        self%subsample_value = settings%subsample
        self%colsample_bytree_value = settings%colsample_bytree
        self%seed_value = settings%seed
        self%restore_best_value = settings%restore_best
        if (objective_code == XGB_OBJECTIVE_HUBER) then
            self%objective_parameter = settings%huber_delta
        else if (objective_code == XGB_OBJECTIVE_QUANTILE) then
            self%objective_parameter = settings%quantile_alpha
        else if (objective_code == XGB_OBJECTIVE_TWEEDIE) then
            self%objective_parameter = settings%tweedie_variance_power
        else
            self%objective_parameter = 0.0_dp
        end if
        self%missing_code = missing_code
        self%categorical_policy_code = categorical_policy_code
        self%categorical_max_categories_value = settings%categorical_max_categories
        if (allocated(settings%categorical_features)) then
            allocate(self%categorical_features(size(settings%categorical_features)))
            self%categorical_features = settings%categorical_features
        else
            allocate(self%categorical_features(0))
        end if
        allocate(self%monotone_constraints(n_features))
        self%monotone_constraints = 0
        if (allocated(settings%monotone_constraints)) then
            self%monotone_constraints = settings%monotone_constraints
        end if
        allocate(self%interaction_groups(n_features))
        self%interaction_groups = 0
        if (allocated(settings%interaction_groups)) then
            self%interaction_groups = settings%interaction_groups
        end if
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_fit

    !> Continue a fitted ensemble up to `options%n_estimators`.
    !>
    !> Warm-starting is a structural continuation: the existing trees are
    !> retained byte-for-byte and only the requested suffix is grown.  The
    !> data, objective, tree controls, sampling stream, and regularisation
    !> must agree with the original fit; `n_estimators`, validation policy,
    !> and `restore_best` may change.  Supplying the same sample weights and
    !> ranking groups is the caller's responsibility because fitted models do
    !> not retain training rows or weights.  A target no larger than the
    !> current ensemble is rejected instead of silently refitting.
    subroutine xgb_fit_warm_start(self, x, y, status, options, sample_weight, &
            validation_x, validation_y, validation_weight, group, validation_group)
        class(xgboost_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_y(:), &
            validation_weight(:)
        integer, intent(in), optional :: group(:), validation_group(:)
        type(xgboost_options_t) :: settings
        real(dp), allocatable :: prediction(:), gradient(:), hessian(:), correction(:)
        real(dp), allocatable :: validation_prediction(:), validation_correction(:)
        real(dp), allocatable :: observation_weight(:), validation_observation_weight(:)
        type(xgb_tree_t), allocatable :: expanded_estimators(:), best_estimators(:)
        integer, allocatable :: sample_index(:)
        logical, allocatable :: feature_mask(:)
        integer :: objective_code, tree_method_code, missing_code, categorical_policy_code
        integer :: n_samples, n_features, n_validation, start_estimators
        integer :: target_estimators, i, completed_estimators, best_iteration, stale_rounds
        integer(int64) :: sampling_state
        real(dp) :: weight_sum, validation_loss, best_validation_loss
        logical :: have_validation, improved, is_ranking, early_stop

        if (.not. self%initialized .or. .not. allocated(self%estimators) .or. &
            .not. allocated(self%monotone_constraints) .or. &
            .not. allocated(self%interaction_groups) .or. &
            .not. allocated(self%categorical_features)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: source is not a valid fitted model")
            return
        end if
        if (.not. present(options)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: options with a larger n_estimators is required")
            return
        end if
        settings = options
        target_estimators = settings%n_estimators
        start_estimators = self%n_estimators
        if (target_estimators <= start_estimators .or. &
            size(self%estimators) /= start_estimators .or. start_estimators < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: n_estimators must exceed the fitted prefix")
            return
        end if

        objective_code = parse_objective(settings%objective)
        tree_method_code = parse_tree_method(settings%tree_method)
        missing_code = parse_missing_policy(settings%missing_policy)
        categorical_policy_code = parse_categorical_policy(settings%categorical_policy)
        if (categorical_policy_code < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: categorical_policy must be none or ordered")
            return
        end if
        is_ranking = self%objective_code == XGB_OBJECTIVE_RANK_PAIRWISE
        if (objective_code /= self%objective_code .or. tree_method_code /= self%tree_method_code .or. &
            missing_code /= self%missing_code .or. settings%max_bin /= self%max_bin .or. &
            settings%max_depth /= self%max_depth_value .or. &
            settings%min_samples_leaf /= self%min_samples_leaf_value .or. &
            settings%learning_rate /= self%learning_rate .or. settings%l1 /= self%l1_value .or. &
            settings%l2 /= self%l2_value .or. settings%gamma /= self%gamma_value .or. &
            settings%min_child_weight /= self%min_child_weight_value .or. &
            settings%subsample /= self%subsample_value .or. &
            settings%colsample_bytree /= self%colsample_bytree_value .or. &
            settings%seed /= self%seed_value .or. categorical_policy_code /= self%categorical_policy_code .or. &
            settings%categorical_max_categories /= self%categorical_max_categories_value) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: objective, tree, sampling, or regularisation controls differ")
            return
        end if
        if (categorical_policy_code == XGB_CATEGORICAL_ORDERED) then
            if (.not. allocated(settings%categorical_features) .or. &
                size(settings%categorical_features) /= size(self%categorical_features) .or. &
                any(settings%categorical_features /= self%categorical_features)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost warm start: categorical feature metadata differs from the fitted model")
                return
            end if
        else if (size(self%categorical_features) /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: categorical policy differs from the fitted model")
            return
        end if
        if (objective_code == 0 .or. tree_method_code == 0 .or. missing_code < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: unsupported objective, tree method, or missing policy")
            return
        end if
        if (objective_code == XGB_OBJECTIVE_TWEEDIE .and. &
            (.not. ieee_is_finite(settings%tweedie_variance_power) .or. &
             settings%tweedie_variance_power <= 1.0_dp .or. &
             settings%tweedie_variance_power >= 2.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: tweedie_variance_power must lie strictly between one and two")
            return
        end if
        if (allocated(settings%monotone_constraints)) then
            if (size(settings%monotone_constraints) /= self%n_inputs .or. &
                any(settings%monotone_constraints /= self%monotone_constraints)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost warm start: monotone constraints differ from the fitted model")
                return
            end if
        else if (any(self%monotone_constraints /= 0)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: monotone constraints must be supplied unchanged")
            return
        end if
        if (allocated(settings%interaction_groups)) then
            if (size(settings%interaction_groups) /= self%n_inputs .or. &
                any(settings%interaction_groups < 0)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost warm start: interaction groups are invalid")
                return
            end if
            if (any(settings%interaction_groups /= self%interaction_groups)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost warm start: interaction groups differ from the fitted model")
                return
            end if
        else if (any(self%interaction_groups /= 0)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: interaction groups must be supplied unchanged")
            return
        end if
        if (objective_code == XGB_OBJECTIVE_HUBER .and. &
            settings%huber_delta /= self%objective_parameter) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: huber_delta differs from the fitted model")
            return
        end if
        if (objective_code == XGB_OBJECTIVE_QUANTILE .and. &
            settings%quantile_alpha /= self%objective_parameter) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: quantile_alpha differs from the fitted model")
            return
        end if
        if (objective_code == XGB_OBJECTIVE_TWEEDIE .and. &
            settings%tweedie_variance_power /= self%objective_parameter) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: tweedie_variance_power differs from the fitted model")
            return
        end if

        have_validation = present(validation_x) .or. present(validation_y) .or. &
            present(validation_weight)
        if (present(validation_x) .neqv. present(validation_y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: validation_x and validation_y must be supplied together")
            return
        end if
        if (present(validation_weight) .and. .not. present(validation_x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: validation_weight requires validation data")
            return
        end if
        if (settings%early_stopping_rounds < 0 .or. &
            .not. ieee_is_finite(settings%early_stopping_min_delta) .or. &
            settings%early_stopping_min_delta < 0.0_dp .or. &
            (settings%early_stopping_rounds > 0 .and. .not. have_validation)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: invalid early-stopping configuration")
            return
        end if
        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (n_samples < 2 .or. n_features /= self%n_inputs .or. size(y) /= n_samples .or. &
            settings%max_depth < 1 .or. settings%max_depth > n_samples .or. &
            settings%min_samples_leaf < 1 .or. 2*settings%min_samples_leaf > n_samples .or. &
            settings%max_bin < 2 .and. tree_method_code == XGB_TREE_HIST) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: input dimensions or tree controls are invalid")
            return
        end if
        if (have_validation) then
            n_validation = size(validation_x, 1)
            if (n_validation < 1 .or. size(validation_x, 2) /= n_features .or. &
                size(validation_y) /= n_validation) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost warm start: validation dimensions do not match training features")
                return
            end if
        else
            n_validation = 0
        end if
        if (is_ranking .neqv. present(group)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: rank:pairwise requires group IDs")
            return
        end if
        if (is_ranking .and. have_validation .and. .not. present(validation_group)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: ranking validation requires validation group IDs")
            return
        end if
        if (is_ranking) then
            if (.not. valid_group_ids(group, n_samples)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost warm start: group IDs are invalid")
                return
            end if
            if (have_validation .and. .not. valid_group_ids(validation_group, n_validation)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost warm start: validation group IDs are invalid")
                return
            end if
        end if
        if (any((.not. ieee_is_finite(x)) .and. (.not. ieee_is_nan(x))) .or. &
            any(.not. ieee_is_finite(y)) .or. &
            (missing_code == XGB_MISSING_ERROR .and. any(ieee_is_nan(x)))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: training inputs are invalid")
            return
        end if
        if (categorical_policy_code == XGB_CATEGORICAL_ORDERED) then
            if (.not. valid_categorical_values(x, settings%categorical_features, &
                    settings%categorical_max_categories)) then
                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                    "xgboost warm start: categorical values exceed integer/max-category boundary")
                return
            end if
            if (have_validation) then
                if (.not. valid_categorical_values(validation_x, settings%categorical_features, &
                        settings%categorical_max_categories)) then
                    call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                        "xgboost warm start: validation categorical values exceed integer/max-category boundary")
                    return
                end if
            end if
        end if
        if (have_validation) then
            if (any((.not. ieee_is_finite(validation_x)) .and. &
                (.not. ieee_is_nan(validation_x))) .or. any(.not. ieee_is_finite(validation_y)) .or. &
                (missing_code == XGB_MISSING_ERROR .and. any(ieee_is_nan(validation_x)))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost warm start: validation inputs are invalid")
                return
            end if
        end if
        if (self%objective_code == XGB_OBJECTIVE_LOGISTIC) then
            if (any(y < 0.0_dp) .or. any(y > 1.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost warm start: logistic targets must be in [0, 1]")
                return
            end if
            if (have_validation .and. (any(validation_y < 0.0_dp) .or. &
                any(validation_y > 1.0_dp))) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost warm start: validation logistic targets must be in [0, 1]")
                return
            end if
        else if (self%objective_code == XGB_OBJECTIVE_POISSON .or. &
            self%objective_code == XGB_OBJECTIVE_TWEEDIE .or. &
            self%objective_code == XGB_OBJECTIVE_SQUARED_LOG) then
            if (any(y < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost warm start: target values must be nonnegative")
                return
            end if
            if (have_validation .and. any(validation_y < 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost warm start: validation target values must be nonnegative")
                return
            end if
        end if

        allocate(observation_weight(n_samples))
        observation_weight = 1.0_dp
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost warm start: sample_weight must be positive and finite")
                return
            end if
            observation_weight = sample_weight
        end if
        weight_sum = sum(observation_weight)
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost warm start: sample_weight has no positive mass")
            return
        end if
        if (have_validation) then
            allocate(validation_observation_weight(n_validation))
            validation_observation_weight = 1.0_dp
            if (present(validation_weight)) then
                if (size(validation_weight) /= n_validation .or. &
                    any(.not. ieee_is_finite(validation_weight)) .or. &
                    any(validation_weight <= 0.0_dp)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "xgboost warm start: validation_weight must be positive and finite")
                    return
                end if
                validation_observation_weight = validation_weight
            end if
        end if

        allocate(prediction(n_samples), correction(n_samples), gradient(n_samples), &
            hessian(n_samples))
        prediction = self%base_score
        do i = 1, start_estimators
            call tree_predict(self%estimators(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            prediction = prediction + self%learning_rate*correction
        end do
        if (have_validation) then
            allocate(validation_prediction(n_validation), validation_correction(n_validation))
            validation_prediction = self%base_score
            do i = 1, start_estimators
                call tree_predict(self%estimators(i), validation_x, validation_correction, status)
                if (status%code /= FORTNUM_OK) return
                validation_prediction = validation_prediction + self%learning_rate*validation_correction
            end do
            call xgb_objective_loss(self%objective_code, validation_prediction, validation_y, &
                validation_observation_weight, self%objective_parameter, &
                merge(self%objective_parameter, 0.5_dp, self%objective_code == XGB_OBJECTIVE_QUANTILE), &
                self%objective_parameter, &
                best_validation_loss, status, validation_group)
            if (status%code /= FORTNUM_OK) return
            best_iteration = start_estimators
            stale_rounds = 0
        else
            best_validation_loss = huge(1.0_dp)
            best_iteration = target_estimators
            stale_rounds = 0
        end if

        allocate(expanded_estimators(target_estimators))
        expanded_estimators(:start_estimators) = self%estimators
        if (have_validation .and. settings%restore_best) best_estimators = self%estimators
        sampling_state = settings%seed
        do i = 1, start_estimators
            call sample_training_rows(n_samples, settings%subsample, sampling_state, sample_index)
            call sample_training_features(n_features, settings%colsample_bytree, sampling_state, feature_mask)
        end do
        completed_estimators = start_estimators
        early_stop = .false.
        do i = start_estimators + 1, target_estimators
            call objective_derivatives(self%objective_code, prediction, y, gradient, hessian, &
                merge(self%objective_parameter, 1.0_dp, self%objective_code == XGB_OBJECTIVE_HUBER), &
                merge(self%objective_parameter, 0.5_dp, self%objective_code == XGB_OBJECTIVE_QUANTILE), &
                self%objective_parameter, &
                status, group, observation_weight)
            if (status%code /= FORTNUM_OK) return
            if (is_ranking) hessian = max(hessian, 1.0e-12_dp)
            if (.not. is_ranking) then
                gradient = observation_weight*gradient
                hessian = observation_weight*hessian
            end if
            call sample_training_rows(n_samples, settings%subsample, sampling_state, sample_index)
            call sample_training_features(n_features, settings%colsample_bytree, sampling_state, feature_mask)
            call build_tree(x, gradient, hessian, observation_weight, settings, sample_index, &
                feature_mask, expanded_estimators(i), status)
            if (status%code /= FORTNUM_OK) return
            call tree_predict(expanded_estimators(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            prediction = prediction + self%learning_rate*correction
            completed_estimators = i
            if (have_validation) then
                call tree_predict(expanded_estimators(i), validation_x, validation_correction, status)
                if (status%code /= FORTNUM_OK) return
                validation_prediction = validation_prediction + self%learning_rate*validation_correction
                call xgb_objective_loss(self%objective_code, validation_prediction, validation_y, &
                    validation_observation_weight, self%objective_parameter, &
                    merge(self%objective_parameter, 0.5_dp, self%objective_code == XGB_OBJECTIVE_QUANTILE), &
                    self%objective_parameter, &
                    validation_loss, status, validation_group)
                if (status%code /= FORTNUM_OK) return
                improved = validation_loss < best_validation_loss - settings%early_stopping_min_delta
                if (improved) then
                    best_validation_loss = validation_loss
                    best_iteration = i
                    stale_rounds = 0
                    if (settings%restore_best) best_estimators = expanded_estimators(:i)
                else
                    stale_rounds = stale_rounds + 1
                end if
                if (settings%early_stopping_rounds > 0 .and. &
                    stale_rounds >= settings%early_stopping_rounds) then
                    early_stop = .true.
                    exit
                end if
            end if
        end do

        if (have_validation .and. settings%restore_best .and. allocated(best_estimators)) then
            call move_alloc(best_estimators, expanded_estimators)
            completed_estimators = size(expanded_estimators)
        else if (completed_estimators < target_estimators) then
            block
                type(xgb_tree_t), allocatable :: retained_estimators(:)
                allocate(retained_estimators(completed_estimators))
                retained_estimators = expanded_estimators(:completed_estimators)
                call move_alloc(retained_estimators, expanded_estimators)
            end block
        end if
        call move_alloc(expanded_estimators, self%estimators)
        self%n_estimators = completed_estimators
        self%requested_estimators = target_estimators
        self%early_stopping_rounds_value = settings%early_stopping_rounds
        self%early_stopping_min_delta_value = settings%early_stopping_min_delta
        self%restore_best_value = settings%restore_best
        self%best_iteration_value = min(max(best_iteration, 1), completed_estimators)
        self%best_validation_loss_value = best_validation_loss
        self%early_stopped_flag = early_stop
        self%initialized = .true.
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_fit_warm_start

    subroutine xgb_fit_regression(self, x, y, status, options, sample_weight, &
            validation_x, validation_y, validation_weight)
        class(xgboost_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_y(:), &
            validation_weight(:)
        type(xgboost_options_t) :: settings

        settings = xgboost_options_t()
        if (present(options)) settings = options
        settings%objective = "squared"
        if (present(validation_x) .or. present(validation_y)) then
            if (present(sample_weight)) then
                if (present(validation_weight)) then
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y, validation_weight)
                else
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y)
                end if
            else if (present(validation_weight)) then
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y, &
                    validation_weight=validation_weight)
            else
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y)
            end if
        else if (present(validation_weight)) then
            call xgb_fit(self, x, y, status, settings, &
                validation_weight=validation_weight)
        else if (present(sample_weight)) then
            call xgb_fit(self, x, y, status, settings, sample_weight)
        else
            call xgb_fit(self, x, y, status, settings)
        end if
    end subroutine xgb_fit_regression

    subroutine xgb_fit_binary(self, x, y, status, options, sample_weight, &
            validation_x, validation_y, validation_weight)
        class(xgboost_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_y(:), &
            validation_weight(:)
        type(xgboost_options_t) :: settings

        settings = xgboost_options_t()
        if (present(options)) settings = options
        settings%objective = "logistic"
        if (present(validation_x) .or. present(validation_y)) then
            if (present(sample_weight)) then
                if (present(validation_weight)) then
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y, validation_weight)
                else
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y)
                end if
            else if (present(validation_weight)) then
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y, &
                    validation_weight=validation_weight)
            else
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y)
            end if
        else if (present(validation_weight)) then
            call xgb_fit(self, x, y, status, settings, &
                validation_weight=validation_weight)
        else if (present(sample_weight)) then
            call xgb_fit(self, x, y, status, settings, sample_weight)
        else
            call xgb_fit(self, x, y, status, settings)
        end if
    end subroutine xgb_fit_binary

    subroutine xgb_fit_poisson(self, x, y, status, options, sample_weight, &
            validation_x, validation_y, validation_weight)
        !! Fit a nonnegative-count model with the log-link Poisson objective.
        !!
        !! The fitted margin is the log mean and predictions are positive
        !! expected counts. Exact and weighted-histogram growth use stable
        !! gradients `exp(margin)-target` and Hessians `exp(margin)`. Input
        !! products remain piecewise-constant tree products; CUDA prediction
        !! is a typed refusal until a resident tree kernel is linked.
        class(xgboost_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_y(:), &
            validation_weight(:)
        type(xgboost_options_t) :: settings

        settings = xgboost_options_t()
        if (present(options)) settings = options
        settings%objective = "poisson"
        if (present(validation_x) .or. present(validation_y)) then
            if (present(sample_weight)) then
                if (present(validation_weight)) then
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y, validation_weight)
                else
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y)
                end if
            else if (present(validation_weight)) then
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y, &
                    validation_weight=validation_weight)
            else
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y)
            end if
        else if (present(validation_weight)) then
            call xgb_fit(self, x, y, status, settings, &
                validation_weight=validation_weight)
        else if (present(sample_weight)) then
            call xgb_fit(self, x, y, status, settings, sample_weight)
        else
            call xgb_fit(self, x, y, status, settings)
        end if
    end subroutine xgb_fit_poisson

    subroutine xgb_fit_tweedie(self, x, y, status, options, sample_weight, &
            validation_x, validation_y, validation_weight)
        !! Fit the bounded XGBoost `reg:tweedie` objective.
        !!
        !! Margins are log means and predictions use the positive inverse link
        !! `exp(margin)`.  The compound-Poisson variance-power regime is
        !! intentionally explicit: `1 < tweedie_variance_power < 2`.
        !! Gradients and Hessians are analytic and finite-guarded.  Tree
        !! fitting remains CPU-only; CUDA prediction is a typed refusal until
        !! a resident tree kernel is linked.
        class(xgboost_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_y(:), &
            validation_weight(:)
        type(xgboost_options_t) :: settings

        settings = xgboost_options_t()
        if (present(options)) settings = options
        settings%objective = "tweedie"
        if (present(validation_x) .or. present(validation_y)) then
            if (present(sample_weight)) then
                if (present(validation_weight)) then
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y, validation_weight)
                else
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y)
                end if
            else if (present(validation_weight)) then
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y, &
                    validation_weight=validation_weight)
            else
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y)
            end if
        else if (present(validation_weight)) then
            call xgb_fit(self, x, y, status, settings, &
                validation_weight=validation_weight)
        else if (present(sample_weight)) then
            call xgb_fit(self, x, y, status, settings, sample_weight)
        else
            call xgb_fit(self, x, y, status, settings)
        end if
    end subroutine xgb_fit_tweedie

    subroutine xgb_fit_squared_log(self, x, y, status, options, sample_weight, &
            validation_x, validation_y, validation_weight)
        !! Fit the XGBoost `reg:squaredlogerror` objective (RMSLE loss).
        !!
        !! Margins represent `log(1 + prediction)`.  Targets must be
        !! nonnegative and public predictions apply the inverse link
        !! `expm1(margin)`.  The exact CPU tree path uses the analytic
        !! gradient and a positive-clipped analytic Hessian.  CUDA prediction
        !! remains a typed refusal until a resident tree kernel is linked.
        class(xgboost_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_y(:), &
            validation_weight(:)
        type(xgboost_options_t) :: settings

        settings = xgboost_options_t()
        if (present(options)) settings = options
        settings%objective = "squaredlog"
        if (present(validation_x) .or. present(validation_y)) then
            if (present(sample_weight)) then
                if (present(validation_weight)) then
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y, validation_weight)
                else
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y)
                end if
            else if (present(validation_weight)) then
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y, &
                    validation_weight=validation_weight)
            else
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y)
            end if
        else if (present(validation_weight)) then
            call xgb_fit(self, x, y, status, settings, &
                validation_weight=validation_weight)
        else if (present(sample_weight)) then
            call xgb_fit(self, x, y, status, settings, sample_weight)
        else
            call xgb_fit(self, x, y, status, settings)
        end if
    end subroutine xgb_fit_squared_log

    subroutine xgb_fit_absolute(self, x, y, status, options, sample_weight, &
            validation_x, validation_y, validation_weight)
        !! Fit an absolute-deviation (L1) regression tree ensemble.
        !!
        !! The margin uses the identity link and the weighted-median constant
        !! initializer.  The exact subgradient is `sign(margin-target)` with
        !! zero at an exact match; a positive Hessian floor keeps the existing
        !! second-order split and leaf machinery well-defined.  The fit and
        !! split decisions are piecewise/discrete, so input products keep the
        !! established split-boundary refusal contract.
        class(xgboost_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_y(:), &
            validation_weight(:)
        type(xgboost_options_t) :: settings

        settings = xgboost_options_t()
        if (present(options)) settings = options
        settings%objective = "absolute"
        if (present(validation_x) .or. present(validation_y)) then
            if (present(sample_weight)) then
                if (present(validation_weight)) then
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y, validation_weight)
                else
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y)
                end if
            else if (present(validation_weight)) then
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y, &
                    validation_weight=validation_weight)
            else
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y)
            end if
        else if (present(validation_weight)) then
            call xgb_fit(self, x, y, status, settings, &
                validation_weight=validation_weight)
        else if (present(sample_weight)) then
            call xgb_fit(self, x, y, status, settings, sample_weight)
        else
            call xgb_fit(self, x, y, status, settings)
        end if
    end subroutine xgb_fit_absolute

    subroutine xgb_fit_huber(self, x, y, status, options, sample_weight, &
            validation_x, validation_y, validation_weight)
        !! Fit a robust Huber regression tree ensemble.  The objective uses
        !! the exact piecewise Huber gradient and a positive Hessian floor on
        !! its linear tails; the split/tree boundary remains nondifferentiable.
        class(xgboost_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_y(:), &
            validation_weight(:)
        type(xgboost_options_t) :: settings

        settings = xgboost_options_t()
        if (present(options)) settings = options
        settings%objective = "huber"
        if (present(validation_x) .or. present(validation_y)) then
            if (present(sample_weight)) then
                if (present(validation_weight)) then
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y, validation_weight)
                else
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y)
                end if
            else if (present(validation_weight)) then
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y, &
                    validation_weight=validation_weight)
            else
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y)
            end if
        else if (present(validation_weight)) then
            call xgb_fit(self, x, y, status, settings, &
                validation_weight=validation_weight)
        else if (present(sample_weight)) then
            call xgb_fit(self, x, y, status, settings, sample_weight)
        else
            call xgb_fit(self, x, y, status, settings)
        end if
    end subroutine xgb_fit_huber

    subroutine xgb_fit_quantile(self, x, y, status, options, sample_weight, &
            validation_x, validation_y, validation_weight)
        !! Fit a pinball/quantile regression tree ensemble.  The subgradient
        !! convention is alpha at an exact zero residual and alpha-1 below
        !! zero; a positive Hessian floor makes Newton leaf updates explicit.
        class(xgboost_t), intent(out) :: self
        real(dp), intent(in) :: x(:, :), y(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :), validation_y(:), &
            validation_weight(:)
        type(xgboost_options_t) :: settings

        settings = xgboost_options_t()
        if (present(options)) settings = options
        settings%objective = "quantile"
        if (present(validation_x) .or. present(validation_y)) then
            if (present(sample_weight)) then
                if (present(validation_weight)) then
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y, validation_weight)
                else
                    call xgb_fit(self, x, y, status, settings, sample_weight, &
                        validation_x, validation_y)
                end if
            else if (present(validation_weight)) then
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y, &
                    validation_weight=validation_weight)
            else
                call xgb_fit(self, x, y, status, settings, &
                    validation_x=validation_x, validation_y=validation_y)
            end if
        else if (present(validation_weight)) then
            call xgb_fit(self, x, y, status, settings, &
                validation_weight=validation_weight)
        else if (present(sample_weight)) then
            call xgb_fit(self, x, y, status, settings, sample_weight)
        else
            call xgb_fit(self, x, y, status, settings)
        end if
    end subroutine xgb_fit_quantile

    subroutine xgb_predict_matrix(self, x, y, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margin(:)

        if (any(shape(y) /= [size(x, 1), 1])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict: output shape is invalid")
            return
        end if
        allocate(margin(size(x, 1)))
        call xgb_predict_margin_vector(self, x, margin, status)
        if (status%code /= FORTNUM_OK) return
        if (self%objective_code == XGB_OBJECTIVE_LOGISTIC) then
            y(:, 1) = stable_sigmoid_array(margin)
        else if (self%objective_code == XGB_OBJECTIVE_POISSON) then
            y(:, 1) = stable_poisson_array(margin)
        else if (self%objective_code == XGB_OBJECTIVE_TWEEDIE) then
            y(:, 1) = stable_poisson_array(margin)
        else if (self%objective_code == XGB_OBJECTIVE_SQUARED_LOG) then
            y(:, 1) = stable_squared_log_array(margin)
        else
            y(:, 1) = margin
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_matrix

    subroutine xgb_predict_vector(self, x, y, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margin(:)

        if (size(y) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict: output shape is invalid")
            return
        end if
        allocate(margin(size(y)))
        call xgb_predict_margin_vector(self, x, margin, status)
        if (status%code /= FORTNUM_OK) return
        if (self%objective_code == XGB_OBJECTIVE_LOGISTIC) then
            y = stable_sigmoid_array(margin)
        else if (self%objective_code == XGB_OBJECTIVE_POISSON) then
            y = stable_poisson_array(margin)
        else if (self%objective_code == XGB_OBJECTIVE_TWEEDIE) then
            y = stable_poisson_array(margin)
        else if (self%objective_code == XGB_OBJECTIVE_SQUARED_LOG) then
            y = stable_squared_log_array(margin)
        else
            y = margin
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_vector

    subroutine xgb_predict_device(self, device, x, y, status)
        !! Predict through the explicit device control-plane contract.
        !!
        !! CPU dispatch reuses the validated host implementation.  XGBoost
        !! tree growth/prediction has no resident CUDA kernel in this build;
        !! selecting CUDA therefore returns a typed refusal and never times a
        !! hidden host fallback as GPU work.
        class(xgboost_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_vector(x, y, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "xgboost device prediction: no resident CUDA tree kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost device prediction: device kind is invalid")
        end select
    end subroutine xgb_predict_device

    subroutine xgb_predict_device_matrix(self, device, x, y, status)
        class(xgboost_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: y(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: values(:)

        if (any(shape(y) /= [size(x, 1), 1])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost device prediction: matrix output shape is invalid")
            return
        end if
        allocate(values(size(x, 1)))
        call xgb_predict_device(self, device, x, values, status)
        if (status%code /= FORTNUM_OK) return
        y(:, 1) = values
    end subroutine xgb_predict_device_matrix

    logical function xgb_device_supported(self, device_kind) result(supported)
        class(xgboost_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = self%initialized .and. device_kind == FORTML_DEVICE_CPU
    end function xgb_device_supported

    subroutine xgb_predict_margin_matrix(self, x, margin, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margin(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (any(shape(margin) /= [size(x, 1), 1])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_margin: output shape is invalid")
            return
        end if
        call xgb_predict_margin_vector(self, x, margin(:, 1), status)
    end subroutine xgb_predict_margin_matrix

    subroutine xgb_predict_margin_vector(self, x, margin, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margin(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: correction(:)
        integer :: i

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(margin) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_margin: model, input, or output shape is invalid")
            return
        end if
        if (.not. valid_query_values(self%missing_code, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_margin: input has unsupported nonfinite values")
            return
        end if
        if (.not. valid_categorical_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_margin: categorical query values must be integer codes")
            return
        end if
        allocate(correction(size(x, 1)))
        margin = self%base_score
        do i = 1, self%n_estimators
            call tree_predict(self%estimators(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            margin = margin + self%learning_rate*correction
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_margin_vector

    !> Return the cumulative prediction after every boosting stage.
    !>
    !> The second dimension is ordered from the first fitted tree through the
    !> complete ensemble. Regression stages contain margins; logistic stages
    !> contain positive-class probabilities; Poisson and Tweedie stages contain positive
    !> expected counts, matching `predict`.
    subroutine xgb_predict_staged(self, x, staged, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: staged(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margin(:), correction(:)
        integer :: i

        if (.not. self%initialized .or. size(staged, 1) /= size(x, 1) .or. &
            size(staged, 2) /= self%n_estimators) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_staged: model, input, or output shape is invalid")
            return
        end if
        if (.not. valid_query_values(self%missing_code, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_staged: input has unsupported nonfinite values")
            return
        end if
        allocate(margin(size(x, 1)), correction(size(x, 1)))
        margin = self%base_score
        do i = 1, self%n_estimators
            call tree_predict(self%estimators(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            margin = margin + self%learning_rate*correction
            if (self%objective_code == XGB_OBJECTIVE_LOGISTIC) then
                staged(:, i) = stable_sigmoid_array(margin)
            else if (self%objective_code == XGB_OBJECTIVE_POISSON) then
                staged(:, i) = stable_poisson_array(margin)
            else if (self%objective_code == XGB_OBJECTIVE_TWEEDIE) then
                staged(:, i) = stable_poisson_array(margin)
            else if (self%objective_code == XGB_OBJECTIVE_SQUARED_LOG) then
                staged(:, i) = stable_squared_log_array(margin)
            else
                staged(:, i) = margin
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_staged

    !> Return cumulative raw margins after every boosting stage.
    subroutine xgb_predict_staged_margin(self, x, staged, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: staged(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: correction(:)
        integer :: i

        if (.not. self%initialized .or. size(staged, 1) /= size(x, 1) .or. &
            size(staged, 2) /= self%n_estimators) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_staged_margin: model, input, or "// &
                "output shape is invalid")
            return
        end if
        if (.not. valid_query_values(self%missing_code, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_staged_margin: input has unsupported nonfinite values")
            return
        end if
        allocate(correction(size(x, 1)))
        staged = 0.0_dp
        staged(:, 1) = self%base_score
        do i = 1, self%n_estimators
            call tree_predict(self%estimators(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            if (i == 1) then
                staged(:, i) = self%base_score + self%learning_rate*correction
            else
                staged(:, i) = staged(:, i - 1) + self%learning_rate*correction
            end if
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_staged_margin

    !> Return additive raw-margin contributions for the fitted ensemble.
    !>
    !> The first column is the fitted base margin.  Column `i+1` is the
    !> learning-rate-scaled contribution of tree `i`; therefore summing the
    !> second dimension reproduces `predict_margin` exactly (up to rounding).
    !> This is the deterministic tree-contribution contract used by model
    !> explanation and deployment code.  For logistic, Poisson, and
    !> squared-log objectives the columns are still raw-link contributions;
    !> apply the objective link only after summing them.
    subroutine xgb_predict_contributions(self, x, contributions, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: contributions(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: correction(:)
        integer :: i

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(contributions, 1) /= size(x, 1) .or. &
            size(contributions, 2) /= self%n_estimators + 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_contributions: model, input, or output "// &
                "shape is invalid")
            return
        end if
        if (.not. valid_query_values(self%missing_code, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_contributions: input has unsupported nonfinite values")
            return
        end if
        if (.not. valid_categorical_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_contributions: categorical query values must be integer codes")
            return
        end if

        allocate(correction(size(x, 1)))
        contributions = 0.0_dp
        contributions(:, 1) = self%base_score
        do i = 1, self%n_estimators
            call tree_predict(self%estimators(i), x, correction, status)
            if (status%code /= FORTNUM_OK) return
            contributions(:, i + 1) = self%learning_rate*correction
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_contributions

    !> Device-control-plane wrapper for additive margin contributions.
    !> CUDA remains a typed refusal until a resident tree kernel is linked.
    subroutine xgb_predict_contributions_device(self, device, x, contributions, &
            status)
        class(xgboost_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: contributions(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost contribution device: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_contributions(x, contributions, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "xgboost contribution device: no resident CUDA tree kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost contribution device: device kind is invalid")
        end select
    end subroutine xgb_predict_contributions_device

    !> Return cumulative binary probabilities after every boosting stage.
    !>
    !> The output has shape `(n_samples, 2, n_estimators)` and stores negative
    !> and positive class probabilities in the second dimension.  A regression
    !> estimator refuses this classification-only operation.
    subroutine xgb_predict_proba_staged(self, x, probabilities, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margins(:, :)
        real(dp) :: positive
        integer :: i, j

        if (self%objective_code /= XGB_OBJECTIVE_LOGISTIC) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_proba_staged: only logistic objective "// &
                "has probabilities")
            return
        end if
        if (size(probabilities, 1) /= size(x, 1) .or. &
            size(probabilities, 2) /= 2 .or. &
            size(probabilities, 3) /= self%n_estimators) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_proba_staged: output shape is invalid")
            return
        end if
        allocate(margins(size(x, 1), self%n_estimators))
        call self%predict_staged_margin(x, margins, status)
        if (status%code /= FORTNUM_OK) return
        do j = 1, self%n_estimators
            do i = 1, size(x, 1)
                positive = stable_sigmoid(margins(i, j))
                probabilities(i, 1, j) = 1.0_dp - positive
                probabilities(i, 2, j) = positive
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_proba_staged

    subroutine xgb_predict_proba(self, x, probabilities, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: margin(:), positive(:)

        if (self%objective_code /= XGB_OBJECTIVE_LOGISTIC) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_proba: only logistic objective has probabilities")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), 2])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_proba: output shape is invalid")
            return
        end if
        allocate(margin(size(x, 1)), positive(size(x, 1)))
        call xgb_predict_margin_vector(self, x, margin, status)
        if (status%code /= FORTNUM_OK) return
        positive = stable_sigmoid_array(margin)
        probabilities(:, 1) = 1.0_dp - positive
        probabilities(:, 2) = positive
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_proba

    !> Input JVP of the piecewise-constant fitted predictor.
    !>
    !> The derivative is zero away from split surfaces.  A query exactly on a
    !> learned threshold is rejected because the tree has no classical
    !> derivative there.
    subroutine xgb_predict_jvp(self, x, x_dot, y, y_dot, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: y(:), y_dot(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, node

        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            any(shape(x_dot) /= shape(x)) .or. size(y) /= size(x, 1) .or. &
            size(y_dot) /= size(y)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_jvp: model, input, or output shape is invalid")
            return
        end if
        if (self%categorical_policy_code == XGB_CATEGORICAL_ORDERED) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "xgboost predict_jvp: categorical feature tangents are discrete")
            return
        end if
        if (.not. valid_query_values(self%missing_code, x) .or. &
            any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_jvp: input or tangent has unsupported values")
            return
        end if
        do j = 1, self%n_estimators
            do node = 1, self%estimators(j)%n_nodes
                if (self%estimators(j)%leaf(node)) cycle
                do i = 1, size(x, 1)
                    if (.not. ieee_is_nan(x(i, self%estimators(j)%feature(node))) .and. &
                        x(i, self%estimators(j)%feature(node)) == &
                        self%estimators(j)%node_threshold(node)) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "xgboost predict_jvp: derivative is undefined on split")
                        return
                    end if
                end do
            end do
        end do
        call xgb_predict_vector(self, x, y, status)
        if (status%code /= FORTNUM_OK) return
        y_dot = 0.0_dp
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_jvp

    subroutine xgb_decision_function(self, x, margin, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margin(:)
        type(fortnum_status_t), intent(out) :: status

        call xgb_predict_margin_vector(self, x, margin, status)
    end subroutine xgb_decision_function

    !> Reverse-mode product of the fitted predictor with respect to query
    !> features.  A fitted tree is piecewise constant, so the product is zero
    !> away from learned split surfaces.  A query exactly on any threshold is
    !> rejected because the classical derivative is undefined there.
    subroutine xgb_predict_vjp(self, x, output_bar, x_bar, status)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), output_bar(:)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, j, node

        x_bar = 0.0_dp
        if (.not. self%initialized .or. size(x, 2) /= self%n_inputs .or. &
            size(output_bar) /= size(x, 1) .or. any(shape(x_bar) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_vjp: model, cotangent, or output shape is invalid")
            return
        end if
        if (self%categorical_policy_code == XGB_CATEGORICAL_ORDERED) then
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "xgboost predict_vjp: categorical feature cotangents are discrete")
            return
        end if
        if (.not. valid_query_values(self%missing_code, x) .or. &
            any(.not. ieee_is_finite(output_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost predict_vjp: input or cotangent has unsupported values")
            return
        end if
        do j = 1, self%n_estimators
            do node = 1, self%estimators(j)%n_nodes
                if (self%estimators(j)%leaf(node)) cycle
                do i = 1, size(x, 1)
                    if (.not. ieee_is_nan(x(i, self%estimators(j)%feature(node))) .and. &
                        x(i, self%estimators(j)%feature(node)) == &
                        self%estimators(j)%node_threshold(node)) then
                        call status_set(status, FORTNUM_DOMAIN_ERROR, &
                            "xgboost predict_vjp: derivative is undefined on split")
                        return
                    end if
                end do
            end do
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_predict_vjp

    !> Copy the first `n_trees` fitted boosting rounds into a valid model.
    !>
    !> Slicing is a structural operation on a fitted ensemble: it does not
    !> refit, rescale, or recompute any tree.  Objective/link, base margin,
    !> regularisation, missing-value routing, constraints, and device metadata
    !> are copied exactly.  Validation diagnostics are retained, while the
    !> reported best iteration is clamped to the retained prefix so the result
    !> remains internally consistent.  A non-fitted source, invalid prefix, or
    !> malformed source is refused without mutating the destination.
    subroutine xgb_slice(self, n_trees, destination, status)
        class(xgboost_t), intent(in) :: self
        integer, intent(in) :: n_trees
        type(xgboost_t), intent(inout) :: destination
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_t) :: candidate
        integer :: i

        if (.not. self%initialized .or. .not. allocated(self%estimators) .or. &
            .not. allocated(self%monotone_constraints) .or. &
            .not. allocated(self%interaction_groups) .or. &
            .not. allocated(self%categorical_features)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost slice: source is not a valid fitted ensemble")
            return
        end if
        if (self%n_estimators < 1 .or. size(self%estimators) /= self%n_estimators .or. &
            size(self%monotone_constraints) /= self%n_inputs .or. &
            size(self%interaction_groups) /= self%n_inputs .or. &
            n_trees < 1 .or. n_trees > self%n_estimators) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost slice: requested prefix is invalid")
            return
        end if
        do i = 1, self%n_estimators
            if (.not. valid_serialized_tree(self%estimators(i), self%n_inputs)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost slice: source contains an invalid tree")
                return
            end if
        end do

        candidate%n_inputs = self%n_inputs
        candidate%n_estimators = n_trees
        candidate%requested_estimators = max(self%requested_estimators, n_trees)
        candidate%objective_code = self%objective_code
        candidate%tree_method_code = self%tree_method_code
        candidate%max_bin = self%max_bin
        candidate%max_depth_value = self%max_depth_value
        candidate%min_samples_leaf_value = self%min_samples_leaf_value
        candidate%early_stopping_rounds_value = self%early_stopping_rounds_value
        candidate%learning_rate = self%learning_rate
        candidate%l1_value = self%l1_value
        candidate%l2_value = self%l2_value
        candidate%gamma_value = self%gamma_value
        candidate%min_child_weight_value = self%min_child_weight_value
        candidate%early_stopping_min_delta_value = self%early_stopping_min_delta_value
        candidate%subsample_value = self%subsample_value
        candidate%colsample_bytree_value = self%colsample_bytree_value
        candidate%seed_value = self%seed_value
        candidate%restore_best_value = self%restore_best_value
        candidate%base_score = self%base_score
        candidate%objective_parameter = self%objective_parameter
        candidate%best_iteration_value = min(max(self%best_iteration_value, 0), n_trees)
        candidate%best_validation_loss_value = self%best_validation_loss_value
        candidate%early_stopped_flag = self%early_stopped_flag
        candidate%missing_code = self%missing_code
        candidate%categorical_policy_code = self%categorical_policy_code
        candidate%categorical_max_categories_value = self%categorical_max_categories_value
        allocate(candidate%monotone_constraints(self%n_inputs))
        candidate%monotone_constraints = self%monotone_constraints
        allocate(candidate%interaction_groups(self%n_inputs))
        candidate%interaction_groups = self%interaction_groups
        allocate(candidate%categorical_features(size(self%categorical_features)))
        candidate%categorical_features = self%categorical_features
        allocate(candidate%estimators(n_trees))
        candidate%estimators = self%estimators(:n_trees)
        candidate%initialized = .true.
        destination = candidate
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_slice

    real(dp) function xgb_split_gain(self, tree_index) result(gain)
        class(xgboost_t), intent(in) :: self
        integer, intent(in) :: tree_index

        gain = 0.0_dp
        if (.not. allocated(self%estimators)) return
        if (tree_index < 1 .or. tree_index > size(self%estimators)) return
        gain = self%estimators(tree_index)%split_gain
    end function xgb_split_gain

    subroutine xgb_leaf_weights(self, tree_index, weights, status)
        class(xgboost_t), intent(in) :: self
        integer, intent(in) :: tree_index
        real(dp), intent(out) :: weights(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. allocated(self%estimators)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost leaf_weights: model is not initialized")
            return
        end if
        if (size(weights) /= 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost leaf_weights: output shape is invalid")
            return
        end if
        if (tree_index < 1 .or. tree_index > size(self%estimators)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost leaf_weights: tree index is invalid")
            return
        end if
        if (self%estimators(tree_index)%n_nodes < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost leaf_weights: tree is not initialized")
            return
        end if
        if (self%estimators(tree_index)%has_split) then
            weights = [self%estimators(tree_index)%weight( &
                self%estimators(tree_index)%left_child(1)), &
                self%estimators(tree_index)%weight( &
                self%estimators(tree_index)%right_child(1))]
        else
            weights = [self%estimators(tree_index)%weight(1), &
                self%estimators(tree_index)%weight(1)]
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_leaf_weights

    !> Aggregate split diagnostics by input feature.
    !>
    !> `kind` is `gain` (the sum of regularized split gains), `weight` (the
    !> number of internal split nodes), or `cover` (the sum of Hessians at
    !> internal nodes).  These are raw sums by default; pass `normalize=.true.`
    !> to divide by the total, as in scikit-learn's feature-importances API.
    subroutine xgb_feature_importance(self, importance, status, kind, normalize)
        class(xgboost_t), intent(in) :: self
        real(dp), intent(out) :: importance(:)
        type(fortnum_status_t), intent(out) :: status
        character(len=*), intent(in), optional :: kind
        logical, intent(in), optional :: normalize
        character(len=16) :: metric
        logical :: should_normalize
        real(dp) :: total
        integer :: tree_index, node, feature

        if (.not. self%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost feature_importance: model is not initialized")
            return
        end if
        if (size(importance) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost feature_importance: output shape is invalid")
            return
        end if
        metric = "gain"
        if (present(kind)) metric = trim(adjustl(kind))
        select case (metric)
        case ("gain", "weight", "cover")
            continue
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost feature_importance: kind must be gain, weight, or cover")
            return
        end select
        should_normalize = .false.
        if (present(normalize)) should_normalize = normalize
        importance = 0.0_dp
        do tree_index = 1, self%n_estimators
            do node = 1, self%estimators(tree_index)%n_nodes
                if (self%estimators(tree_index)%leaf(node)) cycle
                feature = self%estimators(tree_index)%feature(node)
                select case (metric)
                case ("gain")
                    importance(feature) = importance(feature) + &
                        self%estimators(tree_index)%node_gain(node)
                case ("weight")
                    importance(feature) = importance(feature) + 1.0_dp
                case ("cover")
                    importance(feature) = importance(feature) + &
                        self%estimators(tree_index)%node_cover(node)
                end select
            end do
        end do
        if (should_normalize) then
            total = sum(importance)
            if (total > 0.0_dp) importance = importance/total
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_feature_importance

    integer function xgb_tree_node_count(self, tree_index) result(count)
        class(xgboost_t), intent(in) :: self
        integer, intent(in) :: tree_index

        count = 0
        if (.not. allocated(self%estimators)) return
        if (tree_index < 1 .or. tree_index > size(self%estimators)) return
        count = self%estimators(tree_index)%n_nodes
    end function xgb_tree_node_count

    integer function xgb_tree_depth(self, tree_index) result(depth)
        class(xgboost_t), intent(in) :: self
        integer, intent(in) :: tree_index

        depth = 0
        if (.not. allocated(self%estimators)) return
        if (tree_index < 1 .or. tree_index > size(self%estimators)) return
        depth = self%estimators(tree_index)%depth
    end function xgb_tree_depth

    integer function xgb_feature_count(self) result(count)
        class(xgboost_t), intent(in) :: self
        count = self%n_inputs
    end function xgb_feature_count

    integer function xgb_estimator_count(self) result(count)
        class(xgboost_t), intent(in) :: self
        count = self%n_estimators
    end function xgb_estimator_count

    integer function xgb_requested_estimator_count(self) result(count)
        class(xgboost_t), intent(in) :: self

        count = 0
        if (self%initialized) count = self%requested_estimators
    end function xgb_requested_estimator_count

    real(dp) function xgb_base_margin(self) result(margin)
        class(xgboost_t), intent(in) :: self
        margin = self%base_score
    end function xgb_base_margin

    character(len=16) function xgb_objective_name(self) result(name)
        class(xgboost_t), intent(in) :: self

        select case (self%objective_code)
        case (XGB_OBJECTIVE_SQUARED)
            name = "squared"
        case (XGB_OBJECTIVE_LOGISTIC)
            name = "logistic"
        case (XGB_OBJECTIVE_POISSON)
            name = "poisson"
        case (XGB_OBJECTIVE_TWEEDIE)
            name = "tweedie"
        case (XGB_OBJECTIVE_HUBER)
            name = "huber"
        case (XGB_OBJECTIVE_QUANTILE)
            name = "quantile"
        case (XGB_OBJECTIVE_SQUARED_LOG)
            name = "squaredlog"
        case (XGB_OBJECTIVE_RANK_PAIRWISE)
            name = "rank:pairwise"
        case (XGB_OBJECTIVE_ABSOLUTE)
            name = "absolute"
        case default
            name = "unfitted"
        end select
    end function xgb_objective_name

    real(dp) function xgb_objective_parameter(self) result(value)
        class(xgboost_t), intent(in) :: self

        value = self%objective_parameter
    end function xgb_objective_parameter

    character(len=16) function xgb_missing_policy(self) result(name)
        class(xgboost_t), intent(in) :: self

        select case (self%missing_code)
        case (XGB_MISSING_ERROR)
            name = "error"
        case (XGB_MISSING_LEARN)
            name = "learn"
        case (XGB_MISSING_LEFT)
            name = "left"
        case (XGB_MISSING_RIGHT)
            name = "right"
        case default
            name = "unfitted"
        end select
    end function xgb_missing_policy

    character(len=16) function xgb_tree_method(self) result(name)
        class(xgboost_t), intent(in) :: self

        if (.not. self%initialized) then
            name = "unfitted"
            return
        end if
        select case (self%tree_method_code)
        case (XGB_TREE_EXACT)
            name = "exact"
        case (XGB_TREE_HIST)
            name = "hist"
        case default
            name = "unfitted"
        end select
    end function xgb_tree_method

    integer function xgb_max_bin_count(self) result(count)
        class(xgboost_t), intent(in) :: self

        if (self%initialized) then
            count = self%max_bin
        else
            count = 0
        end if
    end function xgb_max_bin_count

    logical function xgb_accepts_missing(self) result(value)
        class(xgboost_t), intent(in) :: self

        value = self%initialized .and. self%missing_code /= XGB_MISSING_ERROR
    end function xgb_accepts_missing

    character(len=16) function xgb_categorical_policy(self) result(name)
        class(xgboost_t), intent(in) :: self

        select case (self%categorical_policy_code)
        case (XGB_CATEGORICAL_NONE)
            name = "none"
        case (XGB_CATEGORICAL_ORDERED)
            name = "ordered"
        case default
            name = "unfitted"
        end select
    end function xgb_categorical_policy

    integer function xgb_categorical_max_categories(self) result(count)
        class(xgboost_t), intent(in) :: self

        count = self%categorical_max_categories_value
    end function xgb_categorical_max_categories

    logical function xgb_categorical_feature(self, feature_index) result(value)
        class(xgboost_t), intent(in) :: self
        integer, intent(in) :: feature_index

        value = .false.
        if (.not. allocated(self%categorical_features)) return
        if (feature_index < 1 .or. feature_index > self%n_inputs) return
        value = any(self%categorical_features == feature_index)
    end function xgb_categorical_feature

    integer function xgb_monotone_constraint(self, feature_index) result(value)
        class(xgboost_t), intent(in) :: self
        integer, intent(in) :: feature_index

        value = 0
        if (.not. self%initialized .or. .not. allocated(self%monotone_constraints)) return
        if (feature_index < 1 .or. feature_index > size(self%monotone_constraints)) return
        value = self%monotone_constraints(feature_index)
    end function xgb_monotone_constraint

    integer function xgb_interaction_group(self, feature_index) result(value)
        !! Return the interaction-group label for one feature. Zero denotes
        !! an unconstrained feature; invalid or unfitted queries return zero.
        class(xgboost_t), intent(in) :: self
        integer, intent(in) :: feature_index

        value = 0
        if (.not. self%initialized .or. .not. allocated(self%interaction_groups)) return
        if (feature_index < 1 .or. feature_index > size(self%interaction_groups)) return
        value = self%interaction_groups(feature_index)
    end function xgb_interaction_group

    logical function xgb_fitted(self) result(fitted)
        class(xgboost_t), intent(in) :: self
        fitted = self%initialized
    end function xgb_fitted

    integer function xgb_best_iteration(self) result(iteration)
        !! One-based boosting round with the lowest validation objective.
        !! Without validation data this is the requested estimator count.
        class(xgboost_t), intent(in) :: self

        iteration = self%best_iteration_value
    end function xgb_best_iteration

    real(dp) function xgb_best_validation_loss(self) result(loss)
        !! Best weighted validation objective observed during fitting.
        !! It is `huge()` when no validation set was supplied.
        class(xgboost_t), intent(in) :: self

        loss = self%best_validation_loss_value
    end function xgb_best_validation_loss

    logical function xgb_early_stopped(self) result(stopped)
        class(xgboost_t), intent(in) :: self

        stopped = self%early_stopped_flag
    end function xgb_early_stopped

    subroutine xgb_save_text(self, path, status)
        !! Save a fitted ensemble as versioned, compiler-independent text.
        !!
        !! The schema uses named records and writes only the live prefix of
        !! every tree node array.  It therefore preserves learned missing
        !! routing, monotone-constrained node values, and validation
        !! diagnostics without exposing private implementation storage.
        class(xgboost_t), intent(in) :: self
        character(*), intent(in) :: path
        type(fortnum_status_t), intent(out) :: status
        integer :: unit, ios, close_ios, i

        if (.not. self%initialized) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost save_text: model is not a valid fitted ensemble")
            return
        end if
        if (.not. allocated(self%estimators)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost save_text: model is not a valid fitted ensemble")
            return
        end if
        if (self%n_estimators < 1 .or. size(self%estimators) /= self%n_estimators) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost save_text: model is not a valid fitted ensemble")
            return
        end if
        if (.not. allocated(self%monotone_constraints)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost save_text: model is not a valid fitted ensemble")
            return
        end if
        if (.not. allocated(self%interaction_groups)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost save_text: model is not a valid fitted ensemble")
            return
        end if
        if (.not. allocated(self%categorical_features)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost save_text: model is not a valid fitted ensemble")
            return
        end if
        if (size(self%monotone_constraints) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost save_text: model is not a valid fitted ensemble")
            return
        end if
        if (size(self%interaction_groups) /= self%n_inputs .or. &
            any(self%interaction_groups < 0)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost save_text: model contains invalid interaction groups")
            return
        end if
        if (self%categorical_policy_code == XGB_CATEGORICAL_ORDERED) then
            if (self%categorical_max_categories_value < 2 .or. &
                size(self%categorical_features) < 1 .or. &
                any(self%categorical_features < 1) .or. &
                any(self%categorical_features > self%n_inputs) .or. &
                has_duplicate_sorted_index(self%categorical_features)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost save_text: model contains invalid categorical metadata")
                return
            end if
        else if (size(self%categorical_features) /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost save_text: categorical feature list requires ordered policy")
            return
        end if
        do i = 1, self%n_estimators
            if (.not. valid_serialized_tree(self%estimators(i), self%n_inputs)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost save_text: model contains an invalid tree")
                return
            end if
        end do

        open(newunit=unit, file=path, status="replace", action="write", &
            form="formatted", access="sequential", iostat=ios)
        if (ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost save_text: cannot open destination")
            return
        end if
        write(unit, "(A)", iostat=ios) XGB_MODEL_TEXT_MAGIC
        if (ios == 0) call xgb_write_i(unit, "schema_version", &
            XGB_MODEL_TEXT_SCHEMA_VERSION, ios)
        if (ios == 0) call xgb_write_i(unit, "n_inputs", self%n_inputs, ios)
        if (ios == 0) call xgb_write_i(unit, "n_estimators", self%n_estimators, ios)
        if (ios == 0) call xgb_write_i(unit, "requested_estimators", &
            self%requested_estimators, ios)
        if (ios == 0) call xgb_write_i(unit, "objective_code", self%objective_code, ios)
        if (ios == 0) call xgb_write_i(unit, "tree_method_code", &
            self%tree_method_code, ios)
        if (ios == 0) call xgb_write_i(unit, "max_bin", self%max_bin, ios)
        if (ios == 0) call xgb_write_i(unit, "max_depth", self%max_depth_value, ios)
        if (ios == 0) call xgb_write_i(unit, "min_samples_leaf", &
            self%min_samples_leaf_value, ios)
        if (ios == 0) call xgb_write_i(unit, "early_stopping_rounds", &
            self%early_stopping_rounds_value, ios)
        if (ios == 0) call xgb_write_r(unit, "learning_rate", self%learning_rate, ios)
        if (ios == 0) call xgb_write_r(unit, "base_score", self%base_score, ios)
        if (ios == 0) call xgb_write_r(unit, "objective_parameter", &
            self%objective_parameter, ios)
        if (ios == 0) call xgb_write_r(unit, "l1", self%l1_value, ios)
        if (ios == 0) call xgb_write_r(unit, "l2", self%l2_value, ios)
        if (ios == 0) call xgb_write_r(unit, "gamma", self%gamma_value, ios)
        if (ios == 0) call xgb_write_r(unit, "min_child_weight", &
            self%min_child_weight_value, ios)
        if (ios == 0) call xgb_write_r(unit, "early_stopping_min_delta", &
            self%early_stopping_min_delta_value, ios)
        if (ios == 0) call xgb_write_r(unit, "subsample", self%subsample_value, ios)
        if (ios == 0) call xgb_write_r(unit, "colsample_bytree", &
            self%colsample_bytree_value, ios)
        if (ios == 0) call xgb_write_i8(unit, "seed", self%seed_value, ios)
        if (ios == 0) call xgb_write_l(unit, "restore_best", &
            self%restore_best_value, ios)
        if (ios == 0) call xgb_write_i(unit, "missing_code", self%missing_code, ios)
        if (ios == 0) call xgb_write_i(unit, "categorical_policy_code", &
            self%categorical_policy_code, ios)
        if (ios == 0) call xgb_write_i(unit, "categorical_max_categories", &
            self%categorical_max_categories_value, ios)
        if (ios == 0) call xgb_write_i(unit, "best_iteration", &
            self%best_iteration_value, ios)
        if (ios == 0) call xgb_write_r(unit, "best_validation_loss", &
            self%best_validation_loss_value, ios)
        if (ios == 0) call xgb_write_l(unit, "early_stopped", &
            self%early_stopped_flag, ios)
        if (ios == 0) call xgb_write_i(unit, "monotone_count", &
            size(self%monotone_constraints), ios)
        do i = 1, self%n_inputs
            if (ios /= 0) exit
            call xgb_write_i(unit, "monotone_item", self%monotone_constraints(i), ios)
        end do
        if (ios == 0) call xgb_write_i(unit, "interaction_count", &
            size(self%interaction_groups), ios)
        do i = 1, self%n_inputs
            if (ios /= 0) exit
            call xgb_write_i(unit, "interaction_item", self%interaction_groups(i), ios)
        end do
        if (ios == 0) call xgb_write_i(unit, "categorical_count", &
            size(self%categorical_features), ios)
        do i = 1, size(self%categorical_features)
            if (ios /= 0) exit
            call xgb_write_i(unit, "categorical_item", self%categorical_features(i), ios)
        end do
        if (ios == 0) call xgb_write_i(unit, "tree_count", self%n_estimators, ios)
        do i = 1, self%n_estimators
            if (ios /= 0) exit
            call xgb_write_tree(unit, self%estimators(i), i, ios)
        end do
        if (ios == 0) write(unit, "(A)", iostat=ios) "end"
        close_ios = 0
        close(unit, iostat=close_ios)
        if (ios /= 0 .or. close_ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost save_text: formatted write failed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_save_text

    subroutine xgb_load_text(self, path, status)
        !! Load a complete text snapshot without partial mutation.
        !!
        !! Every record is checked in schema order.  Unknown, truncated,
        !! duplicate, non-finite, or structurally unsafe records are refused
        !! before the destination model is replaced.
        class(xgboost_t), intent(inout) :: self
        character(*), intent(in) :: path
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_t) :: candidate
        character(len=256) :: line
        integer :: unit, ios, close_ios, schema, i, tree_count, monotone_count
        integer :: interaction_count, categorical_count

        open(newunit=unit, file=path, status="old", action="read", &
            form="formatted", access="sequential", iostat=ios)
        if (ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost load_text: cannot open source")
            return
        end if
        read(unit, "(A)", iostat=ios) line
        if (ios /= 0 .or. trim(line) /= XGB_MODEL_TEXT_MAGIC) goto 900
        call xgb_read_i(unit, "schema_version", schema, ios)
        if (ios /= 0 .or. schema /= XGB_MODEL_TEXT_SCHEMA_VERSION) goto 900
        call xgb_read_i(unit, "n_inputs", candidate%n_inputs, ios)
        if (ios == 0) call xgb_read_i(unit, "n_estimators", candidate%n_estimators, ios)
        if (ios == 0) call xgb_read_i(unit, "requested_estimators", &
            candidate%requested_estimators, ios)
        if (ios == 0) call xgb_read_i(unit, "objective_code", candidate%objective_code, ios)
        if (ios == 0) call xgb_read_i(unit, "tree_method_code", &
            candidate%tree_method_code, ios)
        if (ios == 0) call xgb_read_i(unit, "max_bin", candidate%max_bin, ios)
        if (ios == 0) call xgb_read_i(unit, "max_depth", candidate%max_depth_value, ios)
        if (ios == 0) call xgb_read_i(unit, "min_samples_leaf", &
            candidate%min_samples_leaf_value, ios)
        if (ios == 0) call xgb_read_i(unit, "early_stopping_rounds", &
            candidate%early_stopping_rounds_value, ios)
        if (ios == 0) call xgb_read_r(unit, "learning_rate", candidate%learning_rate, ios)
        if (ios == 0) call xgb_read_r(unit, "base_score", candidate%base_score, ios)
        if (ios == 0) call xgb_read_r(unit, "objective_parameter", &
            candidate%objective_parameter, ios)
        if (ios == 0) call xgb_read_r(unit, "l1", candidate%l1_value, ios)
        if (ios == 0) call xgb_read_r(unit, "l2", candidate%l2_value, ios)
        if (ios == 0) call xgb_read_r(unit, "gamma", candidate%gamma_value, ios)
        if (ios == 0) call xgb_read_r(unit, "min_child_weight", &
            candidate%min_child_weight_value, ios)
        if (ios == 0) call xgb_read_r(unit, "early_stopping_min_delta", &
            candidate%early_stopping_min_delta_value, ios)
        if (ios == 0) call xgb_read_r(unit, "subsample", candidate%subsample_value, ios)
        if (ios == 0) call xgb_read_r(unit, "colsample_bytree", &
            candidate%colsample_bytree_value, ios)
        if (ios == 0) call xgb_read_i8(unit, "seed", candidate%seed_value, ios)
        if (ios == 0) call xgb_read_l(unit, "restore_best", &
            candidate%restore_best_value, ios)
        if (ios == 0) call xgb_read_i(unit, "missing_code", candidate%missing_code, ios)
        if (ios == 0) call xgb_read_i(unit, "categorical_policy_code", &
            candidate%categorical_policy_code, ios)
        if (ios == 0) call xgb_read_i(unit, "categorical_max_categories", &
            candidate%categorical_max_categories_value, ios)
        if (ios == 0) call xgb_read_i(unit, "best_iteration", &
            candidate%best_iteration_value, ios)
        if (ios == 0) call xgb_read_r(unit, "best_validation_loss", &
            candidate%best_validation_loss_value, ios)
        if (ios == 0) call xgb_read_l(unit, "early_stopped", &
            candidate%early_stopped_flag, ios)
        if (ios /= 0) goto 900
        if (.not. valid_serialized_scalars(candidate)) goto 900

        call xgb_read_i(unit, "monotone_count", monotone_count, ios)
        if (ios /= 0 .or. monotone_count /= candidate%n_inputs) goto 900
        allocate(candidate%monotone_constraints(monotone_count), stat=ios)
        if (ios /= 0) goto 900
        do i = 1, monotone_count
            call xgb_read_i(unit, "monotone_item", candidate%monotone_constraints(i), ios)
            if (ios /= 0 .or. abs(candidate%monotone_constraints(i)) > 1) goto 900
        end do
        call xgb_read_i(unit, "interaction_count", interaction_count, ios)
        if (ios /= 0 .or. interaction_count /= candidate%n_inputs) goto 900
        allocate(candidate%interaction_groups(interaction_count), stat=ios)
        if (ios /= 0) goto 900
        do i = 1, interaction_count
            call xgb_read_i(unit, "interaction_item", candidate%interaction_groups(i), ios)
            if (ios /= 0 .or. candidate%interaction_groups(i) < 0) goto 900
        end do
        call xgb_read_i(unit, "categorical_count", categorical_count, ios)
        if (ios /= 0 .or. categorical_count < 0 .or. categorical_count > candidate%n_inputs) goto 900
        allocate(candidate%categorical_features(categorical_count), stat=ios)
        if (ios /= 0) goto 900
        do i = 1, categorical_count
            call xgb_read_i(unit, "categorical_item", candidate%categorical_features(i), ios)
            if (ios /= 0) goto 900
        end do
        if (candidate%categorical_policy_code == XGB_CATEGORICAL_ORDERED) then
            if (categorical_count < 1 .or. candidate%categorical_max_categories_value < 2 .or. &
                any(candidate%categorical_features < 1) .or. &
                any(candidate%categorical_features > candidate%n_inputs) .or. &
                has_duplicate_sorted_index(candidate%categorical_features)) goto 900
        else if (categorical_count /= 0) then
            goto 900
        end if
        call xgb_read_i(unit, "tree_count", tree_count, ios)
        if (ios /= 0 .or. tree_count /= candidate%n_estimators) goto 900
        allocate(candidate%estimators(tree_count), stat=ios)
        if (ios /= 0) goto 900
        do i = 1, tree_count
            call xgb_read_tree(unit, candidate%estimators(i), i, candidate%n_inputs, &
                candidate%categorical_max_categories_value, ios)
            if (ios /= 0) goto 900
        end do
        read(unit, "(A)", iostat=ios) line
        if (ios /= 0 .or. trim(line) /= "end") goto 900
        read(unit, "(A)", iostat=ios) line
        if (ios /= iostat_end) goto 900
        close_ios = 0
        close(unit, iostat=close_ios)
        if (close_ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost load_text: malformed, truncated, or unsupported snapshot")
            return
        end if
        candidate%initialized = .true.
        select type(destination => self)
        type is (xgboost_t)
            destination = candidate
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost load_text: destination type is unsupported")
            return
        end select
        call status_set(status, FORTNUM_OK, "")
        return

        900     continue
        close_ios = 0
        close(unit, iostat=close_ios)
        call status_set(status, FORTNUM_DOMAIN_ERROR, &
            "xgboost load_text: malformed, truncated, or unsupported snapshot")
    end subroutine xgb_load_text

    subroutine xgb_write_tree(unit, tree, index, ios)
        integer, intent(in) :: unit, index
        type(xgb_tree_t), intent(in) :: tree
        integer, intent(out) :: ios
        integer :: i, j

        call xgb_write_i(unit, "tree_begin", index, ios)
        if (ios == 0) call xgb_write_i(unit, "n_nodes", tree%n_nodes, ios)
        if (ios == 0) call xgb_write_i(unit, "depth", tree%depth, ios)
        if (ios == 0) call xgb_write_i(unit, "feature_index", tree%feature_index, ios)
        if (ios == 0) call xgb_write_i(unit, "left_count", tree%left_count, ios)
        if (ios == 0) call xgb_write_i(unit, "right_count", tree%right_count, ios)
        if (ios == 0) call xgb_write_r(unit, "threshold", tree%threshold, ios)
        if (ios == 0) call xgb_write_r(unit, "left_weight", tree%left_weight, ios)
        if (ios == 0) call xgb_write_r(unit, "right_weight", tree%right_weight, ios)
        if (ios == 0) call xgb_write_r(unit, "split_gain", tree%split_gain, ios)
        if (ios == 0) call xgb_write_l(unit, "has_split", tree%has_split, ios)
        if (ios == 0) call xgb_write_i(unit, "node_count", tree%n_nodes, ios)
        do i = 1, tree%n_nodes
            if (ios /= 0) exit
            call xgb_write_i(unit, "node_index", i, ios)
            if (ios == 0) call xgb_write_i(unit, "feature", tree%feature(i), ios)
            if (ios == 0) call xgb_write_i(unit, "left_child", tree%left_child(i), ios)
            if (ios == 0) call xgb_write_i(unit, "right_child", tree%right_child(i), ios)
            if (ios == 0) call xgb_write_r(unit, "node_threshold", &
                tree%node_threshold(i), ios)
            if (ios == 0) call xgb_write_r(unit, "weight", tree%weight(i), ios)
            if (ios == 0) call xgb_write_r(unit, "node_gain", tree%node_gain(i), ios)
            if (ios == 0) call xgb_write_r(unit, "node_cover", tree%node_cover(i), ios)
            if (ios == 0) call xgb_write_l(unit, "leaf", tree%leaf(i), ios)
            if (ios == 0) call xgb_write_l(unit, "missing_left", tree%missing_left(i), ios)
            if (ios == 0) call xgb_write_l(unit, "categorical", tree%categorical(i), ios)
            if (ios == 0) call xgb_write_i(unit, "category_count", tree%category_count(i), ios)
            if (tree%categorical(i)) then
                do j = 1, tree%category_count(i)
                    if (ios /= 0) exit
                    call xgb_write_i(unit, "category_value", tree%category_values(i, j), ios)
                end do
            end if
        end do
        if (ios == 0) write(unit, "(A)", iostat=ios) "tree_end"
    end subroutine xgb_write_tree

    subroutine xgb_read_tree(unit, tree, index, n_inputs, category_capacity, ios)
        integer, intent(in) :: unit, index, n_inputs, category_capacity
        type(xgb_tree_t), intent(out) :: tree
        integer, intent(out) :: ios
        integer :: tree_index, node_count, i, j, alloc_ios, category_count
        character(len=256) :: line_for_tree_record

        call xgb_read_i(unit, "tree_begin", tree_index, ios)
        if (ios /= 0 .or. tree_index /= index) return
        call xgb_read_i(unit, "n_nodes", tree%n_nodes, ios)
        if (ios == 0) call xgb_read_i(unit, "depth", tree%depth, ios)
        if (ios == 0) call xgb_read_i(unit, "feature_index", tree%feature_index, ios)
        if (ios == 0) call xgb_read_i(unit, "left_count", tree%left_count, ios)
        if (ios == 0) call xgb_read_i(unit, "right_count", tree%right_count, ios)
        if (ios == 0) call xgb_read_r(unit, "threshold", tree%threshold, ios)
        if (ios == 0) call xgb_read_r(unit, "left_weight", tree%left_weight, ios)
        if (ios == 0) call xgb_read_r(unit, "right_weight", tree%right_weight, ios)
        if (ios == 0) call xgb_read_r(unit, "split_gain", tree%split_gain, ios)
        if (ios == 0) call xgb_read_l(unit, "has_split", tree%has_split, ios)
        call xgb_read_i(unit, "node_count", node_count, ios)
        if (ios /= 0 .or. node_count /= tree%n_nodes .or. &
            tree%n_nodes < 1 .or. tree%n_nodes > XGB_MAX_SERIALIZED_NODES) then
            ios = 1
            return
        end if
        allocate(tree%feature(tree%n_nodes), tree%left_child(tree%n_nodes), &
            tree%right_child(tree%n_nodes), tree%node_threshold(tree%n_nodes), &
            tree%weight(tree%n_nodes), tree%node_gain(tree%n_nodes), &
            tree%node_cover(tree%n_nodes), tree%leaf(tree%n_nodes), &
            tree%missing_left(tree%n_nodes), tree%categorical(tree%n_nodes), &
            tree%category_count(tree%n_nodes), tree%category_values(tree%n_nodes, max(1, category_capacity)), stat=alloc_ios)
        if (alloc_ios /= 0) then
            ios = 1
            return
        end if
        do i = 1, tree%n_nodes
            call xgb_read_i(unit, "node_index", node_count, ios)
            if (ios /= 0 .or. node_count /= i) return
            call xgb_read_i(unit, "feature", tree%feature(i), ios)
            if (ios == 0) call xgb_read_i(unit, "left_child", tree%left_child(i), ios)
            if (ios == 0) call xgb_read_i(unit, "right_child", tree%right_child(i), ios)
            if (ios == 0) call xgb_read_r(unit, "node_threshold", &
                tree%node_threshold(i), ios)
            if (ios == 0) call xgb_read_r(unit, "weight", tree%weight(i), ios)
            if (ios == 0) call xgb_read_r(unit, "node_gain", tree%node_gain(i), ios)
            if (ios == 0) call xgb_read_r(unit, "node_cover", tree%node_cover(i), ios)
            if (ios == 0) call xgb_read_l(unit, "leaf", tree%leaf(i), ios)
            if (ios == 0) call xgb_read_l(unit, "missing_left", tree%missing_left(i), ios)
            if (ios == 0) call xgb_read_l(unit, "categorical", tree%categorical(i), ios)
            if (ios == 0) call xgb_read_i(unit, "category_count", category_count, ios)
            tree%category_count(i) = category_count
            if (ios == 0 .and. (category_count < 0 .or. category_count > category_capacity)) then
                ios = 1
            end if
            if (ios == 0 .and. tree%categorical(i)) then
                if (category_count < 1) then
                    ios = 1
                else
                    do j = 1, category_count
                        call xgb_read_i(unit, "category_value", tree%category_values(i, j), ios)
                        if (ios /= 0) exit
                    end do
                end if
            end if
            if (ios /= 0) return
        end do
        read(unit, "(A)", iostat=ios) line_for_tree_record
        if (ios /= 0 .or. trim(line_for_tree_record) /= "tree_end") then
            ios = 1
            return
        end if
        if (.not. valid_serialized_tree(tree, n_inputs)) ios = 1
    end subroutine xgb_read_tree

    logical function valid_serialized_scalars(model) result(valid)
        type(xgboost_t), intent(in) :: model

        valid = model%n_inputs >= 1 .and. model%n_estimators >= 1 .and. &
            model%requested_estimators >= model%n_estimators .and. &
            model%objective_code >= XGB_OBJECTIVE_SQUARED .and. &
            model%objective_code <= XGB_OBJECTIVE_TWEEDIE .and. &
            (model%tree_method_code == XGB_TREE_EXACT .or. &
             model%tree_method_code == XGB_TREE_HIST) .and. &
            model%max_bin >= 2 .and. model%max_depth_value >= 1 .and. &
            model%min_samples_leaf_value >= 1
        if (.not. valid) return
        valid = model%early_stopping_rounds_value >= 0 .and. &
            model%early_stopping_rounds_value >= 0 .and. &
            model%missing_code >= XGB_MISSING_ERROR .and. &
            model%missing_code <= XGB_MISSING_RIGHT .and. &
            model%best_iteration_value >= 0 .and. &
            model%best_iteration_value <= model%requested_estimators .and. &
            model%seed_value > 0_int64 .and. &
            ieee_is_finite(model%learning_rate) .and. model%learning_rate > 0.0_dp .and. &
            model%learning_rate <= 1.0_dp .and. ieee_is_finite(model%base_score) .and. &
            ieee_is_finite(model%objective_parameter) .and. ieee_is_finite(model%l1_value) .and. &
            model%l1_value >= 0.0_dp .and. ieee_is_finite(model%l2_value) .and. &
            model%l2_value >= 0.0_dp .and. ieee_is_finite(model%gamma_value) .and. &
            model%gamma_value >= 0.0_dp .and. ieee_is_finite(model%min_child_weight_value) .and. &
            model%min_child_weight_value >= 0.0_dp .and. &
            ieee_is_finite(model%early_stopping_min_delta_value) .and. &
            model%early_stopping_min_delta_value >= 0.0_dp .and. &
            ieee_is_finite(model%subsample_value) .and. model%subsample_value > 0.0_dp .and. &
            model%subsample_value <= 1.0_dp .and. ieee_is_finite(model%colsample_bytree_value) .and. &
            model%colsample_bytree_value > 0.0_dp .and. model%colsample_bytree_value <= 1.0_dp .and. &
            ieee_is_finite(model%best_validation_loss_value)
        if (valid) then
            valid = (model%categorical_policy_code == XGB_CATEGORICAL_NONE .and. &
                model%categorical_max_categories_value >= 0) .or. &
                (model%categorical_policy_code == XGB_CATEGORICAL_ORDERED .and. &
                model%categorical_max_categories_value >= 2)
        end if
        if (valid .and. model%objective_code == XGB_OBJECTIVE_TWEEDIE) then
            valid = model%objective_parameter > 1.0_dp .and. &
                model%objective_parameter < 2.0_dp
        end if
    end function valid_serialized_scalars

    logical function valid_serialized_tree(tree, n_inputs) result(valid)
        type(xgb_tree_t), intent(in) :: tree
        integer, intent(in) :: n_inputs
        logical, allocatable :: seen(:)
        integer, allocatable :: stack(:)
        integer :: i, j, node, top

        valid = tree%n_nodes >= 1 .and. tree%n_nodes <= XGB_MAX_SERIALIZED_NODES .and. &
            tree%depth >= 0 .and. tree%feature_index >= 0 .and. &
            tree%feature_index <= n_inputs .and. tree%left_count >= 0 .and. &
            tree%right_count >= 0 .and. ieee_is_finite(tree%threshold) .and. &
            ieee_is_finite(tree%left_weight) .and. ieee_is_finite(tree%right_weight) .and. &
            ieee_is_finite(tree%split_gain) .and. allocated(tree%feature) .and. &
            allocated(tree%left_child) .and. allocated(tree%right_child) .and. &
            allocated(tree%node_threshold) .and. allocated(tree%weight) .and. &
            allocated(tree%node_gain) .and. allocated(tree%node_cover) .and. &
            allocated(tree%leaf) .and. allocated(tree%missing_left) .and. &
            allocated(tree%categorical) .and. allocated(tree%category_count) .and. &
            allocated(tree%category_values)
        if (.not. valid) return
        valid = size(tree%feature) >= tree%n_nodes .and. &
            size(tree%left_child) >= tree%n_nodes .and. &
            size(tree%right_child) >= tree%n_nodes .and. &
            size(tree%node_threshold) >= tree%n_nodes .and. &
            size(tree%weight) >= tree%n_nodes .and. &
            size(tree%node_gain) >= tree%n_nodes .and. &
            size(tree%node_cover) >= tree%n_nodes .and. &
            size(tree%leaf) >= tree%n_nodes .and. &
            size(tree%missing_left) >= tree%n_nodes .and. &
            size(tree%categorical) >= tree%n_nodes .and. &
            size(tree%category_count) >= tree%n_nodes .and. &
            size(tree%category_values, 1) >= tree%n_nodes
        if (.not. valid) return
        if (any(.not. ieee_is_finite(tree%node_threshold)) .or. &
            any(.not. ieee_is_finite(tree%weight)) .or. &
            any(.not. ieee_is_finite(tree%node_gain)) .or. &
            any(.not. ieee_is_finite(tree%node_cover))) then
            valid = .false.
            return
        end if
        do i = 1, tree%n_nodes
            if (tree%leaf(i)) then
                if (tree%categorical(i) .or. tree%category_count(i) /= 0) then
                    valid = .false.
                    return
                end if
                if (tree%feature(i) /= 0 .or. tree%left_child(i) /= 0 .or. &
                    tree%right_child(i) /= 0) then
                    valid = .false.
                    return
                end if
            else
                if (tree%feature(i) < 1 .or. tree%feature(i) > n_inputs) then
                    valid = .false.
                    return
                end if
                if (tree%category_count(i) < 0 .or. tree%category_count(i) > &
                    size(tree%category_values, 2)) then
                    valid = .false.
                    return
                end if
                if (tree%categorical(i) .and. tree%category_count(i) < 1) then
                    valid = .false.
                    return
                end if
                if (.not. tree%categorical(i) .and. tree%category_count(i) /= 0) then
                    valid = .false.
                    return
                end if
                if (tree%categorical(i)) then
                    if (tree%category_count(i) > 1) then
                        do j = 1, tree%category_count(i) - 1
                            if (any(tree%category_values(i, j + 1:tree%category_count(i)) == &
                                tree%category_values(i, j))) then
                                valid = .false.
                                return
                            end if
                        end do
                    end if
                end if
                if (tree%left_child(i) <= i) then
                    valid = .false.
                    return
                end if
                if (tree%left_child(i) > tree%n_nodes) then
                    valid = .false.
                    return
                end if
                if (tree%right_child(i) <= i) then
                    valid = .false.
                    return
                end if
                if (tree%right_child(i) > tree%n_nodes) then
                    valid = .false.
                    return
                end if
            end if
        end do
        if (tree%has_split .neqv. .not. tree%leaf(1)) then
            valid = .false.
            return
        end if
        allocate(seen(tree%n_nodes), stack(tree%n_nodes))
        seen = .false.
        top = 1
        stack(1) = 1
        do while (top > 0)
            node = stack(top)
            top = top - 1
            if (seen(node)) cycle
            seen(node) = .true.
            if (.not. tree%leaf(node)) then
                top = top + 1
                stack(top) = tree%left_child(node)
                top = top + 1
                stack(top) = tree%right_child(node)
            end if
        end do
        valid = all(seen)
    end function valid_serialized_tree

    subroutine xgb_write_i(unit, key, value, ios)
        integer, intent(in) :: unit, value
        character(*), intent(in) :: key
        integer, intent(out) :: ios

        write(unit, "(A,1X,I0)", iostat=ios) trim(key), value
    end subroutine xgb_write_i

    subroutine xgb_write_i8(unit, key, value, ios)
        integer, intent(in) :: unit
        integer(int64), intent(in) :: value
        character(*), intent(in) :: key
        integer, intent(out) :: ios

        write(unit, "(A,1X,I0)", iostat=ios) trim(key), value
    end subroutine xgb_write_i8

    subroutine xgb_write_l(unit, key, value, ios)
        integer, intent(in) :: unit
        logical, intent(in) :: value
        character(*), intent(in) :: key
        integer, intent(out) :: ios

        write(unit, "(A,1X,I0)", iostat=ios) trim(key), merge(1, 0, value)
    end subroutine xgb_write_l

    subroutine xgb_write_r(unit, key, value, ios)
        integer, intent(in) :: unit
        real(dp), intent(in) :: value
        character(*), intent(in) :: key
        integer, intent(out) :: ios

        write(unit, "(A,1X,ES26.17E3)", iostat=ios) trim(key), value
    end subroutine xgb_write_r

    subroutine xgb_read_i(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        integer, intent(out) :: value
        integer, intent(out) :: ios
        character(len=80) :: key

        read(unit, *, iostat=ios) key, value
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
    end subroutine xgb_read_i

    subroutine xgb_read_i8(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        integer(int64), intent(out) :: value
        integer, intent(out) :: ios
        character(len=80) :: key

        read(unit, *, iostat=ios) key, value
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
    end subroutine xgb_read_i8

    subroutine xgb_read_l(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        logical, intent(out) :: value
        integer, intent(out) :: ios
        character(len=80) :: key
        integer :: encoded

        encoded = 0
        read(unit, *, iostat=ios) key, encoded
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
        if (ios == 0 .and. encoded /= 0 .and. encoded /= 1) ios = 1
        value = encoded == 1
    end subroutine xgb_read_l

    subroutine xgb_read_r(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        real(dp), intent(out) :: value
        integer, intent(out) :: ios
        character(len=80) :: key

        read(unit, *, iostat=ios) key, value
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
    end subroutine xgb_read_r

    subroutine retain_xgb_estimators(self, source)
        class(xgboost_t), intent(inout) :: self
        type(xgb_tree_t), intent(in) :: source(:)
        type(xgb_tree_t), allocatable :: retained(:)

        allocate(retained(size(source)))
        retained = source
        call move_alloc(retained, self%estimators)
    end subroutine retain_xgb_estimators

    !> Evaluate the normalized pairwise logistic ranking loss.
    !! Only pairs with equal query IDs and unequal labels contribute.  For a
    !! preferred row `i` over `j`, the term is
    !! `log(1 + exp(-(margin_i-margin_j)))`; pair weights are the smaller of
    !! the two optional observation weights.
    subroutine xgb_pairwise_loss(margin, target, group, loss, status, weights)
        real(dp), intent(in) :: margin(:), target(:)
        integer, intent(in) :: group(:)
        real(dp), intent(out) :: loss
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: weights(:)
        real(dp) :: weight_i, weight_j, pair_weight, delta, term, weight_sum
        integer :: i, j, pair_count

        loss = huge(1.0_dp)
        if (.not. valid_pairwise_arrays(margin, target, group, weights)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost pairwise loss: invalid arrays or weights")
            return
        end if
        term = 0.0_dp
        weight_sum = 0.0_dp
        pair_count = 0
        do i = 1, size(target) - 1
            do j = i + 1, size(target)
                if (group(i) /= group(j) .or. target(i) == target(j)) cycle
                pair_count = pair_count + 1
                if (present(weights)) then
                    weight_i = weights(i)
                    weight_j = weights(j)
                else
                    weight_i = 1.0_dp
                    weight_j = 1.0_dp
                end if
                pair_weight = min(weight_i, weight_j)
                if (target(i) > target(j)) then
                    delta = margin(i) - margin(j)
                else
                    delta = margin(j) - margin(i)
                end if
                if (delta >= 0.0_dp) then
                    term = term + pair_weight*log(1.0_dp + exp(-delta))
                else
                    term = term + pair_weight*(-delta + log(1.0_dp + exp(delta)))
                end if
                weight_sum = weight_sum + pair_weight
            end do
        end do
        if (pair_count < 1 .or. .not. ieee_is_finite(weight_sum) .or. &
            weight_sum <= 0.0_dp .or. .not. ieee_is_finite(term)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost pairwise loss: every query needs an unequal-label pair")
            return
        end if
        loss = term/weight_sum
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_pairwise_loss

    !> Return unnormalised pairwise gradients and positive Hessians suitable
    !! for the XGBoost tree gain formula.  The gradient/Hessian sums use the
    !! same pair weights as `xgb_pairwise_loss`.
    subroutine xgb_pairwise_derivatives(margin, target, group, gradient, hessian, &
            status, weights)
        real(dp), intent(in) :: margin(:), target(:)
        integer, intent(in) :: group(:)
        real(dp), intent(out) :: gradient(:), hessian(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: weights(:)
        real(dp), parameter :: minimum_hessian = 1.0e-12_dp
        real(dp) :: weight_i, weight_j, pair_weight, delta, probability
        integer :: i, j, high, low, pair_count

        gradient = 0.0_dp
        hessian = 0.0_dp
        if (size(gradient) /= size(margin) .or. size(hessian) /= size(margin) .or. &
            .not. valid_pairwise_arrays(margin, target, group, weights)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost pairwise derivatives: invalid arrays or weights")
            return
        end if
        pair_count = 0
        do i = 1, size(target) - 1
            do j = i + 1, size(target)
                if (group(i) /= group(j) .or. target(i) == target(j)) cycle
                pair_count = pair_count + 1
                if (target(i) > target(j)) then
                    high = i
                    low = j
                else
                    high = j
                    low = i
                end if
                if (present(weights)) then
                    weight_i = weights(i)
                    weight_j = weights(j)
                else
                    weight_i = 1.0_dp
                    weight_j = 1.0_dp
                end if
                pair_weight = min(weight_i, weight_j)
                delta = margin(high) - margin(low)
                if (delta >= 0.0_dp) then
                    probability = exp(-delta)/(1.0_dp + exp(-delta))
                else
                    probability = 1.0_dp/(1.0_dp + exp(delta))
                end if
                gradient(high) = gradient(high) - pair_weight*probability
                gradient(low) = gradient(low) + pair_weight*probability
                hessian(high) = hessian(high) + pair_weight*max( &
                    probability*(1.0_dp - probability), minimum_hessian)
                hessian(low) = hessian(low) + pair_weight*max( &
                    probability*(1.0_dp - probability), minimum_hessian)
            end do
        end do
        if (pair_count < 1 .or. any(.not. ieee_is_finite(gradient)) .or. &
            any(.not. ieee_is_finite(hessian))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost pairwise derivatives: every query needs an unequal-label pair")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_pairwise_derivatives

    logical function valid_pairwise_arrays(margin, target, group, weights) result(valid)
        real(dp), intent(in) :: margin(:), target(:)
        integer, intent(in) :: group(:)
        real(dp), intent(in), optional :: weights(:)

        valid = size(margin) >= 2 .and. size(target) == size(margin) .and. &
            size(group) == size(margin) .and. any(group > 0) .and. &
            all(group > 0) .and. all(ieee_is_finite(margin)) .and. &
            all(ieee_is_finite(target))
        if (.not. valid) return
        if (present(weights)) then
            valid = size(weights) == size(margin) .and. &
                all(ieee_is_finite(weights)) .and. all(weights > 0.0_dp)
        end if
    end function valid_pairwise_arrays

    subroutine xgb_tweedie_loss(margin, target, variance_power, value, status, weights)
        !! Return the weighted mean Tweedie negative log-likelihood (up to the
        !! target-only normalization constant) in the log-mean margin.
        !!
        !! For `1 < p < 2`, the per-row value is
        !! `y*exp((1-p)*margin)/(p-1) + exp((2-p)*margin)/(2-p)`.
        !! Optional positive weights are normalized by their total mass.
        real(dp), intent(in) :: margin(:), target(:), variance_power
        real(dp), intent(out) :: value
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in), optional :: weights(:)
        real(dp), allocatable :: row_weight(:)
        real(dp) :: weight_sum, row_value, first_term, second_term
        integer :: i

        value = huge(1.0_dp)
        if (size(margin) < 1 .or. size(target) /= size(margin) .or. &
            .not. ieee_is_finite(variance_power) .or. variance_power <= 1.0_dp .or. &
            variance_power >= 2.0_dp .or. any(.not. ieee_is_finite(margin)) .or. &
            any(.not. ieee_is_finite(target)) .or. any(target < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost Tweedie loss: finite margins, nonnegative targets, and 1 < power < 2 are required")
            return
        end if
        allocate(row_weight(size(target)))
        row_weight = 1.0_dp
        if (present(weights)) then
            if (size(weights) /= size(target) .or. any(.not. ieee_is_finite(weights)) .or. &
                any(weights <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost Tweedie loss: weights must be positive and finite")
                return
            end if
            row_weight = weights
        end if
        weight_sum = sum(row_weight)
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost Tweedie loss: weight mass must be positive and finite")
            return
        end if
        value = 0.0_dp
        do i = 1, size(target)
            first_term = stable_tweedie_exp(margin(i), 1.0_dp - variance_power)
            second_term = stable_tweedie_exp(margin(i), 2.0_dp - variance_power)
            row_value = target(i)*first_term/(variance_power - 1.0_dp) + &
                second_term/(2.0_dp - variance_power)
            if (.not. ieee_is_finite(row_value)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost Tweedie loss: objective value is nonfinite")
                return
            end if
            value = value + row_weight(i)*row_value
        end do
        if (.not. ieee_is_finite(value)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost Tweedie loss: weighted objective is nonfinite")
            return
        end if
        value = value/weight_sum
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_tweedie_loss

    subroutine xgb_tweedie_derivatives(margin, target, variance_power, gradient, &
            hessian, status)
        !! Return exact per-row Tweedie derivatives with respect to log mean.
        !! The Hessian is strictly nonnegative in the supported compound-
        !! Poisson power interval; no artificial floor is included here.
        real(dp), intent(in) :: margin(:), target(:), variance_power
        real(dp), intent(out) :: gradient(:), hessian(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp) :: first_term, second_term
        integer :: i

        if (size(margin) < 1 .or. size(target) /= size(margin) .or. &
            size(gradient) /= size(margin) .or. size(hessian) /= size(margin) .or. &
            .not. ieee_is_finite(variance_power) .or. variance_power <= 1.0_dp .or. &
            variance_power >= 2.0_dp .or. any(.not. ieee_is_finite(margin)) .or. &
            any(.not. ieee_is_finite(target)) .or. any(target < 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost Tweedie derivatives: finite margins, nonnegative targets, and 1 < power < 2 are required")
            return
        end if
        do i = 1, size(margin)
            first_term = stable_tweedie_exp(margin(i), 1.0_dp - variance_power)
            second_term = stable_tweedie_exp(margin(i), 2.0_dp - variance_power)
            gradient(i) = -target(i)*first_term + second_term
            hessian(i) = target(i)*(variance_power - 1.0_dp)*first_term + &
                (2.0_dp - variance_power)*second_term
        end do
        if (any(.not. ieee_is_finite(gradient)) .or. &
            any(.not. ieee_is_finite(hessian))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost Tweedie derivatives: gradient or Hessian is nonfinite")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_tweedie_derivatives

    subroutine xgb_objective_loss(objective_code, margin, target, weights, &
            huber_delta, quantile_alpha, tweedie_power, loss, status, group)
        !! Evaluate a finite weighted validation objective independently of
        !! the tree gain calculation.  All supported objectives are losses,
        !! so lower values are better for deterministic early stopping.
        integer, intent(in) :: objective_code
        real(dp), intent(in) :: margin(:), target(:), weights(:)
        real(dp), intent(in) :: huber_delta, quantile_alpha, tweedie_power
        real(dp), intent(out) :: loss
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: group(:)
        real(dp) :: residual, term, weight_sum
        integer :: i

        loss = huge(1.0_dp)
        if (size(margin) /= size(target) .or. size(weights) /= size(target) .or. &
            size(target) < 1 .or. any(.not. ieee_is_finite(margin)) .or. &
            any(.not. ieee_is_finite(target)) .or. &
            any(.not. ieee_is_finite(weights)) .or. any(weights <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost validation objective: invalid arrays")
            return
        end if
        weight_sum = sum(weights)
        if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost validation objective: invalid weight mass")
            return
        end if
        term = 0.0_dp
        if (objective_code == XGB_OBJECTIVE_RANK_PAIRWISE) then
            if (.not. present(group)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost validation objective: ranking group IDs are missing")
                return
            end if
            call xgb_pairwise_loss(margin, target, group, loss, status, weights)
            return
        end if
        select case (objective_code)
        case (XGB_OBJECTIVE_SQUARED)
            term = 0.5_dp*sum(weights*(margin - target)**2)
        case (XGB_OBJECTIVE_LOGISTIC)
            do i = 1, size(target)
                if (margin(i) >= 0.0_dp) then
                    residual = (1.0_dp - target(i))*margin(i) + &
                        log(1.0_dp + exp(-margin(i)))
                else
                    residual = -target(i)*margin(i) + log(1.0_dp + exp(margin(i)))
                end if
                term = term + weights(i)*residual
            end do
        case (XGB_OBJECTIVE_POISSON)
            term = sum(weights*(stable_poisson_array(margin) - target*margin))
        case (XGB_OBJECTIVE_TWEEDIE)
            call xgb_tweedie_loss(margin, target, tweedie_power, loss, status, weights)
            return
        case (XGB_OBJECTIVE_SQUARED_LOG)
            term = 0.5_dp*sum(weights*(margin - log(1.0_dp + target))**2)
        case (XGB_OBJECTIVE_HUBER)
            do i = 1, size(target)
                residual = margin(i) - target(i)
                if (abs(residual) <= huber_delta) then
                    term = term + weights(i)*0.5_dp*residual**2
                else
                    term = term + weights(i)*huber_delta*(abs(residual) - &
                        0.5_dp*huber_delta)
                end if
            end do
        case (XGB_OBJECTIVE_QUANTILE)
            do i = 1, size(target)
                residual = margin(i) - target(i)
                if (residual >= 0.0_dp) then
                    term = term + weights(i)*quantile_alpha*residual
                else
                    term = term + weights(i)*(quantile_alpha - 1.0_dp)*residual
                end if
            end do
        case (XGB_OBJECTIVE_ABSOLUTE)
            term = sum(weights*abs(margin - target))
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost validation objective: unsupported objective")
            return
        end select
        if (.not. ieee_is_finite(term)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost validation objective: nonfinite loss")
            return
        end if
        loss = term/weight_sum
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_objective_loss

    subroutine objective_derivatives(objective_code, margin, target, gradient, &
            hessian, huber_delta, quantile_alpha, tweedie_power, status, group, weights)
        integer, intent(in) :: objective_code
        real(dp), intent(in) :: margin(:), target(:)
        real(dp), intent(out) :: gradient(:), hessian(:)
        real(dp), intent(in) :: huber_delta, quantile_alpha, tweedie_power
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: group(:)
        real(dp), intent(in), optional :: weights(:)
        real(dp), parameter :: minimum_hessian = 1.0e-12_dp
        real(dp), allocatable :: probability(:)
        real(dp) :: residual, probability_value
        integer :: i

        if (size(margin) /= size(target) .or. size(gradient) /= size(target) .or. &
            size(hessian) /= size(target)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost derivatives: array shapes differ")
            return
        end if
        if (objective_code == XGB_OBJECTIVE_RANK_PAIRWISE) then
            if (.not. present(group)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "xgboost derivatives: ranking group IDs are missing")
                return
            end if
            call xgb_pairwise_derivatives(margin, target, group, gradient, hessian, &
                status, weights)
            return
        end if
        select case (objective_code)
        case (XGB_OBJECTIVE_SQUARED)
            gradient = margin - target
            hessian = 1.0_dp
        case (XGB_OBJECTIVE_LOGISTIC)
            allocate(probability(size(margin)))
            probability = stable_sigmoid_array(margin)
            gradient = probability - target
            hessian = max(probability*(1.0_dp - probability), minimum_hessian)
        case (XGB_OBJECTIVE_POISSON)
            allocate(probability(size(margin)))
            probability = stable_poisson_array(margin)
            gradient = probability - target
            hessian = max(probability, minimum_hessian)
        case (XGB_OBJECTIVE_TWEEDIE)
            call xgb_tweedie_derivatives(margin, target, tweedie_power, gradient, &
                hessian, status)
            if (status%code /= FORTNUM_OK) return
        case (XGB_OBJECTIVE_SQUARED_LOG)
            ! f(m) = 1/2 [m-log(1+y)]^2 after setting prediction=expm1(m).
            ! The chain rule gives g=(m-log1p(y))/exp(m) and
            ! h=(1-(m-log1p(y)))/exp(m).  XGBoost's tree gain formulas
            ! require positive curvature, so retain the standard floor when
            ! the exact Hessian is nonpositive in a far tail.
            do i = 1, size(margin)
                residual = margin(i) - log(1.0_dp + target(i))
                probability_value = stable_poisson_mean(margin(i))
                gradient(i) = residual/probability_value
                hessian(i) = max((1.0_dp - residual)/probability_value, &
                    minimum_hessian)
            end do
        case (XGB_OBJECTIVE_HUBER)
            do i = 1, size(margin)
                residual = margin(i) - target(i)
                if (abs(residual) <= huber_delta) then
                    gradient(i) = residual
                    hessian(i) = 1.0_dp
                else
                    gradient(i) = huber_delta*sign(1.0_dp, residual)
                    hessian(i) = minimum_hessian
                end if
            end do
        case (XGB_OBJECTIVE_QUANTILE)
            do i = 1, size(margin)
                if (margin(i) - target(i) >= 0.0_dp) then
                    gradient(i) = quantile_alpha
                else
                    gradient(i) = quantile_alpha - 1.0_dp
                end if
                hessian(i) = minimum_hessian
            end do
        case (XGB_OBJECTIVE_ABSOLUTE)
            do i = 1, size(margin)
                residual = margin(i) - target(i)
                if (residual > 0.0_dp) then
                    gradient(i) = 1.0_dp
                else if (residual < 0.0_dp) then
                    gradient(i) = -1.0_dp
                else
                    gradient(i) = 0.0_dp
                end if
                hessian(i) = minimum_hessian
            end do
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost derivatives: unsupported objective")
            return
        end select
        if (any(.not. ieee_is_finite(gradient)) .or. &
            any(.not. ieee_is_finite(hessian))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost derivatives: nonfinite gradient or Hessian")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine objective_derivatives

    subroutine sample_training_rows(n_samples, fraction, state, sample_index)
        !! Draw a deterministic without-replacement row subset.  The output
        !! is sorted by original row index so feature-value ties retain the
        !! same stable ordering as the full-data path.
        integer, intent(in) :: n_samples
        real(dp), intent(in) :: fraction
        integer(int64), intent(inout) :: state
        integer, allocatable, intent(out) :: sample_index(:)
        integer, allocatable :: permutation(:)
        logical, allocatable :: selected(:)
        integer :: n_selected, i, j, k, temporary

        if (fraction >= 1.0_dp) then
            allocate(sample_index(n_samples))
            do i = 1, n_samples
                sample_index(i) = i
            end do
            return
        end if
        n_selected = max(1, min(n_samples, int(ceiling(fraction*real(n_samples, dp)))))
        allocate(permutation(n_samples), selected(n_samples), sample_index(n_selected))
        do i = 1, n_samples
            permutation(i) = i
        end do
        do i = 1, n_selected
            j = i + int(mod(next_sampling_integer(state), int(n_samples - i + 1, int64)))
            temporary = permutation(i)
            permutation(i) = permutation(j)
            permutation(j) = temporary
        end do
        selected = .false.
        do i = 1, n_selected
            selected(permutation(i)) = .true.
        end do
        k = 0
        do i = 1, n_samples
            if (selected(i)) then
                k = k + 1
                sample_index(k) = i
            end if
        end do
    end subroutine sample_training_rows

    subroutine sample_training_features(n_features, fraction, state, feature_mask)
        !! Draw a deterministic without-replacement feature subset.  The
        !! mask is consumed in ascending feature order by tree growth.
        integer, intent(in) :: n_features
        real(dp), intent(in) :: fraction
        integer(int64), intent(inout) :: state
        logical, allocatable, intent(out) :: feature_mask(:)
        integer, allocatable :: permutation(:)
        integer :: n_selected, i, j, k, temporary

        allocate(feature_mask(n_features))
        feature_mask = .false.
        if (fraction >= 1.0_dp) then
            feature_mask = .true.
            return
        end if
        n_selected = max(1, min(n_features, int(ceiling(fraction*real(n_features, dp)))))
        allocate(permutation(n_features))
        do i = 1, n_features
            permutation(i) = i
        end do
        do i = 1, n_selected
            j = i + int(mod(next_sampling_integer(state), int(n_features - i + 1, int64)))
            temporary = permutation(i)
            permutation(i) = permutation(j)
            permutation(j) = temporary
        end do
        do k = 1, n_selected
            feature_mask(permutation(k)) = .true.
        end do
    end subroutine sample_training_features

    integer(int64) function next_sampling_integer(state) result(value)
        integer(int64), intent(inout) :: state
        integer(int64), parameter :: modulus = 2147483647_int64
        integer(int64), parameter :: multiplier = 48271_int64

        ! Reduce arbitrary positive user seeds before the multiply so the
        ! fixed-width product cannot overflow int64.
        state = modulo(state, modulus)
        if (state <= 0_int64) state = 1_int64
        state = mod(multiplier*state, modulus)
        if (state <= 0_int64) state = 1_int64
        value = state
    end function next_sampling_integer

    subroutine build_tree(x, gradient, hessian, observation_weight, options, &
            sample_index, feature_mask, tree, status)
        real(dp), intent(in) :: x(:, :), gradient(:), hessian(:)
        real(dp), intent(in) :: observation_weight(:)
        type(xgboost_options_t), intent(in) :: options
        integer, intent(in) :: sample_index(:)
        logical, intent(in) :: feature_mask(:)
        type(xgb_tree_t), intent(out) :: tree
        type(fortnum_status_t), intent(out) :: status
        integer :: n_samples, n_features, max_nodes, next_node, root

        n_samples = size(x, 1)
        n_features = size(x, 2)
        if (size(gradient) /= size(x, 1) .or. size(hessian) /= size(x, 1) .or. &
            size(observation_weight) /= size(x, 1) .or. size(sample_index) < 1 .or. &
            size(feature_mask) /= n_features .or. .not. any(feature_mask)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost tree: derivative array shape is invalid")
            return
        end if
        if (n_samples < 1 .or. n_features < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost tree: at least one sample and feature are required")
            return
        end if

        ! A binary tree containing n leaves has at most 2*n-1 nodes.  This
        ! bound avoids integer exponentiation and is also a hard memory bound
        ! independent of a user-supplied depth.
        n_samples = size(sample_index)
        max_nodes = 2*n_samples - 1
        allocate(tree%feature(max_nodes), tree%left_child(max_nodes), &
            tree%right_child(max_nodes), tree%node_threshold(max_nodes), &
            tree%weight(max_nodes), tree%node_gain(max_nodes), &
            tree%node_cover(max_nodes), &
            tree%leaf(max_nodes), tree%missing_left(max_nodes), &
            tree%categorical(max_nodes), tree%category_count(max_nodes), &
            tree%category_values(max_nodes, max(1, options%categorical_max_categories)))
        tree%feature = 0
        tree%left_child = 0
        tree%right_child = 0
        tree%node_threshold = 0.0_dp
        tree%weight = 0.0_dp
        tree%node_gain = 0.0_dp
        tree%node_cover = 0.0_dp
        tree%leaf = .true.
        tree%missing_left = .true.
        tree%categorical = .false.
        tree%category_count = 0
        tree%category_values = 0
        next_node = 0
        tree%depth = 0
        call build_tree_node(x, gradient, hessian, observation_weight, options, &
            sample_index, feature_mask, feature_mask, 0, tree, next_node, root, status, &
            -huge(1.0_dp), huge(1.0_dp))
        if (status%code /= FORTNUM_OK) return
        tree%n_nodes = next_node
        tree%feature_index = tree%feature(root)
        tree%split_gain = tree%node_gain(root)
        tree%has_split = .not. tree%leaf(root)
        if (tree%has_split) tree%threshold = tree%node_threshold(root)
        tree%left_weight = tree%weight(root)
        tree%right_weight = tree%weight(root)
        if (tree%has_split) then
            tree%left_weight = tree%weight(tree%left_child(root))
            tree%right_weight = tree%weight(tree%right_child(root))
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine build_tree

    recursive subroutine build_tree_node(x, gradient, hessian, observation_weight, &
            options, sample_index, feature_mask, path_feature_mask, depth, tree, &
            next_node, node_id, status, lower_bound, upper_bound)
        real(dp), intent(in) :: x(:, :), gradient(:), hessian(:)
        real(dp), intent(in) :: observation_weight(:)
        type(xgboost_options_t), intent(in) :: options
        integer, intent(in) :: sample_index(:), depth
        logical, intent(in) :: feature_mask(:)
        logical, intent(in) :: path_feature_mask(:)
        type(xgb_tree_t), intent(inout) :: tree
        integer, intent(inout) :: next_node
        integer, intent(out) :: node_id
        type(fortnum_status_t), intent(out) :: status
        real(dp), intent(in) :: lower_bound, upper_bound
        integer, allocatable :: order(:), left_index(:), right_index(:)
        logical, allocatable :: left_path_mask(:), right_path_mask(:)
        integer, allocatable :: finite_index(:), candidate_position(:)
        integer, allocatable :: category_values_local(:), category_counts_local(:)
        integer, allocatable :: category_order(:), best_categories(:)
        logical, allocatable :: candidate_mask(:)
        real(dp), allocatable :: feature_values(:), finite_values(:)
        real(dp), allocatable :: category_gradients(:), category_hessians(:), category_scores(:)
        integer :: n_local, n_features, feature, k, i, left_count
        integer :: n_finite, n_missing, direction, n_directions
        integer :: n_candidates, candidate_index
        integer :: best_feature, left_node, right_node
        real(dp) :: total_gradient, total_hessian, left_gradient, left_hessian
        real(dp) :: right_gradient, right_hessian, candidate_gain, best_gain
        real(dp) :: best_threshold, candidate_threshold, value
        real(dp) :: missing_gradient, missing_hessian
        real(dp) :: candidate_left_weight, candidate_right_weight
        real(dp) :: candidate_bound, candidate_left_lower, candidate_left_upper
        real(dp) :: candidate_right_lower, candidate_right_upper
        real(dp) :: best_left_lower, best_left_upper
        real(dp) :: best_right_lower, best_right_upper
        integer :: monotone
        integer :: n_categories, category_index, best_category_count
        logical :: best_missing_left, missing_left, best_is_categorical

        n_local = size(sample_index)
        n_features = size(x, 2)
        next_node = next_node + 1
        node_id = next_node
        tree%depth = max(tree%depth, depth)
        total_gradient = sum(gradient(sample_index))
        total_hessian = sum(hessian(sample_index))
        value = bounded_leaf_weight(total_gradient, total_hessian, options, &
            lower_bound, upper_bound)
        tree%weight(node_id) = value
        tree%node_gain(node_id) = 0.0_dp
        tree%node_cover(node_id) = total_hessian
        tree%leaf(node_id) = .true.
        tree%missing_left(node_id) = .true.

        ! No split is possible at the depth limit or when both children would
        ! violate the minimum leaf size.  Returning a regularized leaf here is
        ! exactly the XGBoost Newton-leaf contract.
        if (depth >= options%max_depth .or. &
            n_local < 2*options%min_samples_leaf) then
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        allocate(order(n_local), feature_values(n_local), finite_index(n_local), &
            finite_values(n_local))
        best_gain = 0.0_dp
        best_feature = 0
        best_threshold = 0.0_dp
        best_missing_left = .true.
        best_left_lower = lower_bound
        best_left_upper = upper_bound
        best_right_lower = lower_bound
        best_right_upper = upper_bound
        best_is_categorical = .false.
        best_category_count = 0
        allocate(category_values_local(max(1, options%categorical_max_categories)), &
            category_counts_local(max(1, options%categorical_max_categories)), &
            category_gradients(max(1, options%categorical_max_categories)), &
            category_hessians(max(1, options%categorical_max_categories)), &
            category_scores(max(1, options%categorical_max_categories)), &
            category_order(max(1, options%categorical_max_categories)), &
            best_categories(max(1, options%categorical_max_categories)))
        do feature = 1, n_features
            if (.not. feature_mask(feature) .or. &
                .not. path_feature_mask(feature)) cycle
            if (is_categorical_feature(options, feature)) then
                n_categories = 0
                n_missing = 0
                missing_gradient = 0.0_dp
                missing_hessian = 0.0_dp
                category_values_local = 0
                category_counts_local = 0
                category_gradients = 0.0_dp
                category_hessians = 0.0_dp
                do i = 1, n_local
                    value = x(sample_index(i), feature)
                    if (ieee_is_nan(value)) then
                        n_missing = n_missing + 1
                        missing_gradient = missing_gradient + gradient(sample_index(i))
                        missing_hessian = missing_hessian + hessian(sample_index(i))
                    else
                        if (.not. is_integer_code(value)) then
                            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                                "xgboost tree: categorical value is not an integer code")
                            return
                        end if
                        category_index = find_category(category_values_local, n_categories, &
                            nint(value))
                        if (category_index == 0) then
                            if (n_categories >= options%categorical_max_categories) then
                                call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                                    "xgboost tree: categorical cardinality exceeds categorical_max_categories")
                                return
                            end if
                            n_categories = n_categories + 1
                            category_index = n_categories
                            category_values_local(category_index) = nint(value)
                        end if
                        category_counts_local(category_index) = category_counts_local(category_index) + 1
                        category_gradients(category_index) = category_gradients(category_index) + &
                            gradient(sample_index(i))
                        category_hessians(category_index) = category_hessians(category_index) + &
                            hessian(sample_index(i))
                    end if
                end do
                if (n_categories < 2) cycle
                do i = 1, n_categories
                    category_order(i) = i
                    category_scores(i) = category_gradients(i) / max(category_hessians(i), 1.0e-12_dp)
                end do
                call sort_category_order(category_values_local, category_scores, category_order, n_categories)
                left_gradient = 0.0_dp
                left_hessian = 0.0_dp
                left_count = 0
                n_directions = 1
                if (missing_code_for_options(options) == XGB_MISSING_LEARN) n_directions = 2
                do k = 1, n_categories - 1
                    category_index = category_order(k)
                    left_gradient = left_gradient + category_gradients(category_index)
                    left_hessian = left_hessian + category_hessians(category_index)
                    left_count = left_count + category_counts_local(category_index)
                    do direction = 1, n_directions
                        missing_left = direction == 1
                        if (missing_code_for_options(options) == XGB_MISSING_RIGHT) then
                            missing_left = .false.
                        else if (missing_code_for_options(options) == XGB_MISSING_LEFT) then
                            missing_left = .true.
                        end if
                        if (missing_left) then
                            if (left_count + n_missing < options%min_samples_leaf .or. &
                                n_local - left_count < options%min_samples_leaf) cycle
                            if (left_hessian + missing_hessian < options%min_child_weight .or. &
                                total_hessian - left_hessian - missing_hessian < options%min_child_weight) cycle
                            candidate_left_weight = bounded_leaf_weight(left_gradient + missing_gradient, &
                                left_hessian + missing_hessian, options, lower_bound, upper_bound)
                            candidate_right_weight = bounded_leaf_weight(total_gradient - left_gradient - &
                                missing_gradient, total_hessian - left_hessian - missing_hessian, options, &
                                lower_bound, upper_bound)
                            candidate_gain = 0.5_dp*(regularized_leaf_score(left_gradient + missing_gradient, &
                                left_hessian + missing_hessian, options) + regularized_leaf_score( &
                                total_gradient - left_gradient - missing_gradient, total_hessian - &
                                left_hessian - missing_hessian, options) - regularized_leaf_score(total_gradient, &
                                total_hessian, options)) - options%gamma
                        else
                            if (left_count < options%min_samples_leaf .or. &
                                n_local - left_count + n_missing < options%min_samples_leaf) cycle
                            if (left_hessian < options%min_child_weight .or. &
                                total_hessian - left_hessian - missing_hessian < options%min_child_weight) cycle
                            candidate_left_weight = bounded_leaf_weight(left_gradient, left_hessian, options, &
                                lower_bound, upper_bound)
                            candidate_right_weight = bounded_leaf_weight(total_gradient - left_gradient - &
                                missing_gradient, total_hessian - left_hessian - missing_hessian, options, &
                                lower_bound, upper_bound)
                            candidate_gain = 0.5_dp*(regularized_leaf_score(left_gradient, left_hessian, options) + &
                                regularized_leaf_score(total_gradient - left_gradient - missing_gradient, &
                                total_hessian - left_hessian - missing_hessian, options) - regularized_leaf_score( &
                                total_gradient, total_hessian, options)) - options%gamma
                        end if
                        ! Monotonic constraints are defined on ordered numeric cuts;
                        ! categorical partitions retain the unconstrained policy.
                        if (candidate_gain > best_gain) then
                            best_gain = candidate_gain
                            best_feature = feature
                            best_threshold = 0.0_dp
                            best_missing_left = missing_left
                            best_is_categorical = .true.
                            best_category_count = k
                            best_categories(:k) = category_values_local(category_order(:k))
                            best_left_lower = lower_bound
                            best_left_upper = upper_bound
                            best_right_lower = lower_bound
                            best_right_upper = upper_bound
                        end if
                    end do
                end do
                cycle
            end if
            n_finite = 0
            n_missing = 0
            missing_gradient = 0.0_dp
            missing_hessian = 0.0_dp
            do i = 1, n_local
                feature_values(i) = x(sample_index(i), feature)
                if (ieee_is_nan(feature_values(i))) then
                    n_missing = n_missing + 1
                    missing_gradient = missing_gradient + &
                        gradient(sample_index(i))
                    missing_hessian = missing_hessian + hessian(sample_index(i))
                else
                    n_finite = n_finite + 1
                    finite_index(n_finite) = sample_index(i)
                    finite_values(n_finite) = feature_values(i)
                end if
            end do
            if (n_finite < 2) cycle
            call sort_feature_indices(finite_values(:n_finite), order(:n_finite))
            allocate(candidate_position(max(1, n_finite - 1)), &
                candidate_mask(max(1, n_finite - 1)))
            candidate_mask = .false.
            if (parse_tree_method(options%tree_method) == XGB_TREE_HIST) then
                call histogram_cut_positions(finite_values(:n_finite), order(:n_finite), &
                    finite_index(:n_finite), observation_weight, options%max_bin, &
                    candidate_position, n_candidates)
                if (n_candidates > 0) candidate_mask(candidate_position(:n_candidates)) = .true.
            else
                n_candidates = n_finite - 1
                candidate_mask(:n_candidates) = .true.
            end if
            left_gradient = 0.0_dp
            left_hessian = 0.0_dp
            n_directions = 1
            if (missing_code_for_options(options) == XGB_MISSING_LEARN) then
                n_directions = 2
            end if
            do k = 1, n_finite - 1
                i = order(k)
                left_gradient = left_gradient + gradient(finite_index(i))
                left_hessian = left_hessian + hessian(finite_index(i))
                if (.not. candidate_mask(k)) cycle
                if (k < options%min_samples_leaf .or. &
                    n_finite - k + n_missing < options%min_samples_leaf) cycle
                if (finite_values(order(k)) >= finite_values(order(k + 1))) cycle
                candidate_threshold = safe_midpoint(finite_values(order(k)), &
                    finite_values(order(k + 1)))
                do direction = 1, n_directions
                    missing_left = direction == 1
                    if (missing_code_for_options(options) == XGB_MISSING_RIGHT) then
                        missing_left = .false.
                    else if (missing_code_for_options(options) == XGB_MISSING_LEFT) then
                        missing_left = .true.
                    end if
                    if (missing_left) then
                        if (k + n_missing < options%min_samples_leaf .or. &
                            n_finite - k < options%min_samples_leaf) cycle
                    else
                        if (k < options%min_samples_leaf .or. &
                            n_finite - k + n_missing < options%min_samples_leaf) cycle
                    end if
                    if (missing_left) then
                        left_gradient = left_gradient + missing_gradient
                        left_hessian = left_hessian + missing_hessian
                        right_gradient = total_gradient - left_gradient
                        right_hessian = total_hessian - left_hessian
                    else
                        right_gradient = total_gradient - left_gradient - &
                            missing_gradient
                        right_hessian = total_hessian - left_hessian - &
                            missing_hessian
                    end if
                    if (left_hessian < options%min_child_weight .or. &
                        right_hessian < options%min_child_weight) then
                        if (missing_left) then
                            left_gradient = left_gradient - missing_gradient
                            left_hessian = left_hessian - missing_hessian
                        end if
                        cycle
                    end if
                    candidate_left_weight = bounded_leaf_weight(left_gradient, &
                        left_hessian, options, lower_bound, upper_bound)
                    candidate_right_weight = bounded_leaf_weight(right_gradient, &
                        right_hessian, options, lower_bound, upper_bound)
                    monotone = monotone_constraint_for_feature(options, feature)
                    candidate_left_lower = lower_bound
                    candidate_left_upper = upper_bound
                    candidate_right_lower = lower_bound
                    candidate_right_upper = upper_bound
                    if (monotone > 0) then
                        if (candidate_left_weight > candidate_right_weight + &
                            1.0e-14_dp) then
                            if (missing_left) then
                                left_gradient = left_gradient - missing_gradient
                                left_hessian = left_hessian - missing_hessian
                            end if
                            cycle
                        end if
                        candidate_bound = safe_midpoint(candidate_left_weight, &
                            candidate_right_weight)
                        candidate_left_upper = min(candidate_left_upper, candidate_bound)
                        candidate_right_lower = max(candidate_right_lower, candidate_bound)
                    else if (monotone < 0) then
                        if (candidate_left_weight < candidate_right_weight - &
                            1.0e-14_dp) then
                            if (missing_left) then
                                left_gradient = left_gradient - missing_gradient
                                left_hessian = left_hessian - missing_hessian
                            end if
                            cycle
                        end if
                        candidate_bound = safe_midpoint(candidate_left_weight, &
                            candidate_right_weight)
                        candidate_left_lower = max(candidate_left_lower, candidate_bound)
                        candidate_right_upper = min(candidate_right_upper, candidate_bound)
                    end if
                    candidate_gain = 0.5_dp*(regularized_leaf_score(left_gradient, &
                        left_hessian, options) + regularized_leaf_score(right_gradient, &
                        right_hessian, options) - regularized_leaf_score(total_gradient, &
                        total_hessian, options)) - options%gamma
                    if (candidate_gain > best_gain) then
                        best_gain = candidate_gain
                        best_feature = feature
                        best_threshold = candidate_threshold
                        best_missing_left = missing_left
                        best_left_lower = candidate_left_lower
                        best_left_upper = candidate_left_upper
                        best_right_lower = candidate_right_lower
                        best_right_upper = candidate_right_upper
                    end if
                    if (missing_left) then
                        left_gradient = left_gradient - missing_gradient
                        left_hessian = left_hessian - missing_hessian
                    end if
                end do
            end do
            deallocate(candidate_position, candidate_mask)
        end do

        if (best_feature == 0) then
            deallocate(category_values_local, category_counts_local, category_gradients, &
                category_hessians, category_scores, category_order, best_categories)
            call status_set(status, FORTNUM_OK, "")
            return
        end if

        left_count = 0
        do i = 1, n_local
            if (best_is_categorical) then
                if (category_go_left(x(sample_index(i), best_feature), best_categories, &
                        best_category_count, best_missing_left)) left_count = left_count + 1
            else if (go_left(x(sample_index(i), best_feature), best_threshold, &
                    best_missing_left)) then
                left_count = left_count + 1
            end if
        end do
        if (left_count < options%min_samples_leaf .or. &
            n_local - left_count < options%min_samples_leaf) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost tree: split construction violated leaf constraint")
            return
        end if
        allocate(left_index(left_count), right_index(n_local - left_count))
        left_count = 0
        k = 0
        do i = 1, n_local
            if ((best_is_categorical .and. category_go_left(x(sample_index(i), best_feature), &
                    best_categories, best_category_count, best_missing_left)) .or. &
                ((.not. best_is_categorical) .and. go_left(x(sample_index(i), best_feature), &
                    best_threshold, best_missing_left))) then
                left_count = left_count + 1
                left_index(left_count) = sample_index(i)
            else
                k = k + 1
                right_index(k) = sample_index(i)
            end if
        end do
        tree%leaf(node_id) = .false.
        tree%feature(node_id) = best_feature
        tree%node_threshold(node_id) = best_threshold
        tree%node_gain(node_id) = best_gain
        tree%missing_left(node_id) = best_missing_left
        tree%categorical(node_id) = best_is_categorical
        tree%category_count(node_id) = 0
        if (best_is_categorical) then
            tree%category_count(node_id) = best_category_count
            tree%category_values(node_id, :best_category_count) = best_categories(:best_category_count)
        end if
        allocate(left_path_mask(n_features), right_path_mask(n_features))
        left_path_mask = path_feature_mask
        right_path_mask = path_feature_mask
        if (interaction_group_for_feature(options, best_feature) > 0) then
            left_path_mask = .false.
            right_path_mask = .false.
            do i = 1, n_features
                if (feature_mask(i) .and. interaction_group_for_feature(options, i) == &
                    interaction_group_for_feature(options, best_feature)) then
                    left_path_mask(i) = .true.
                    right_path_mask(i) = .true.
                end if
            end do
        end if
        call build_tree_node(x, gradient, hessian, observation_weight, options, &
            left_index, feature_mask, left_path_mask, depth + 1, tree, next_node, &
            left_node, status, best_left_lower, best_left_upper)
        if (status%code /= FORTNUM_OK) return
        call build_tree_node(x, gradient, hessian, observation_weight, options, &
            right_index, feature_mask, right_path_mask, depth + 1, tree, next_node, &
            right_node, status, best_right_lower, best_right_upper)
        deallocate(left_path_mask, right_path_mask)
        deallocate(category_values_local, category_counts_local, category_gradients, &
            category_hessians, category_scores, category_order, best_categories)
        if (status%code /= FORTNUM_OK) return
        tree%left_child(node_id) = left_node
        tree%right_child(node_id) = right_node
        call status_set(status, FORTNUM_OK, "")
    end subroutine build_tree_node

    subroutine tree_predict(tree, x, values, status)
        type(xgb_tree_t), intent(in) :: tree
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: values(:)
        type(fortnum_status_t), intent(out) :: status
        integer :: i, node

        if (size(values) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost tree prediction: output shape is invalid")
            return
        end if
        if (tree%n_nodes < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost tree prediction: tree is not initialized")
            return
        end if
        if (.not. allocated(tree%leaf)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "xgboost tree prediction: tree is not initialized")
            return
        end if
        do i = 1, size(x, 1)
            node = 1
            do while (.not. tree%leaf(node))
                if (tree%categorical(node)) then
                    if (category_go_left(x(i, tree%feature(node)), tree%category_values(node, :), &
                            tree%category_count(node), tree%missing_left(node))) then
                        node = tree%left_child(node)
                    else
                        node = tree%right_child(node)
                    end if
                else if (go_left(x(i, tree%feature(node)), &
                    tree%node_threshold(node), tree%missing_left(node))) then
                    node = tree%left_child(node)
                else
                    node = tree%right_child(node)
                end if
            end do
            values(i) = tree%weight(node)
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine tree_predict

    real(dp) function regularized_leaf_weight(gradient, hessian, options) &
            result(weight)
        real(dp), intent(in) :: gradient, hessian
        type(xgboost_options_t), intent(in) :: options
        real(dp) :: thresholded

        thresholded = max(abs(gradient) - options%l1, 0.0_dp)
        if (thresholded == 0.0_dp) then
            weight = 0.0_dp
        else
            weight = -sign(thresholded, gradient)/(hessian + options%l2)
        end if
    end function regularized_leaf_weight

    real(dp) function bounded_leaf_weight(gradient, hessian, options, lower, upper) &
            result(weight)
        real(dp), intent(in) :: gradient, hessian, lower, upper
        type(xgboost_options_t), intent(in) :: options

        weight = regularized_leaf_weight(gradient, hessian, options)
        weight = min(max(weight, lower), upper)
    end function bounded_leaf_weight

    real(dp) function regularized_leaf_score(gradient, hessian, options) &
            result(score)
        real(dp), intent(in) :: gradient, hessian
        type(xgboost_options_t), intent(in) :: options
        real(dp) :: thresholded

        thresholded = max(abs(gradient) - options%l1, 0.0_dp)
        score = thresholded**2/(hessian + options%l2)
    end function regularized_leaf_score

    integer function parse_objective(name) result(code)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: normalized

        normalized = trim(adjustl(name))
        select case (normalized)
        case ("squared", "reg:squarederror", "regression")
            code = XGB_OBJECTIVE_SQUARED
        case ("logistic", "binary:logistic", "classification")
            code = XGB_OBJECTIVE_LOGISTIC
        case ("poisson", "count:poisson", "reg:poisson")
            code = XGB_OBJECTIVE_POISSON
        case ("tweedie", "reg:tweedie", "reg:tweedieerror")
            code = XGB_OBJECTIVE_TWEEDIE
        case ("huber", "reg:pseudohubererror")
            code = XGB_OBJECTIVE_HUBER
        case ("quantile", "reg:quantile", "pinball")
            code = XGB_OBJECTIVE_QUANTILE
        case ("squaredlog", "squared-log", "squaredlogerror", &
                "reg:squaredlogerror", "rmsle")
            code = XGB_OBJECTIVE_SQUARED_LOG
        case ("rank:pairwise", "rank_pairwise", "pairwise", "ranking")
            code = XGB_OBJECTIVE_RANK_PAIRWISE
        case ("absolute", "reg:absolute", "reg:absoluteerror", "mae", &
                "l1")
            code = XGB_OBJECTIVE_ABSOLUTE
        case default
            code = 0
        end select
    end function parse_objective

    logical function valid_group_ids(group, n_rows) result(valid)
        integer, intent(in), optional :: group(:)
        integer, intent(in) :: n_rows

        valid = present(group)
        if (.not. valid) return
        valid = size(group) == n_rows .and. n_rows >= 2 .and. all(group > 0)
    end function valid_group_ids

    integer function parse_missing_policy(name) result(code)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: normalized

        normalized = trim(adjustl(name))
        select case (normalized)
        case ("error", "finite-only", "reject")
            code = XGB_MISSING_ERROR
        case ("learn", "learned", "default")
            code = XGB_MISSING_LEARN
        case ("left", "missing-left")
            code = XGB_MISSING_LEFT
        case ("right", "missing-right")
            code = XGB_MISSING_RIGHT
        case default
            code = -1
        end select
    end function parse_missing_policy

    integer function parse_tree_method(name) result(code)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: normalized

        normalized = trim(adjustl(name))
        select case (normalized)
        case ("exact", "auto")
            code = XGB_TREE_EXACT
        case ("hist", "histogram", "approx")
            code = XGB_TREE_HIST
        case default
            code = 0
        end select
    end function parse_tree_method

    integer function parse_categorical_policy(name) result(code)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: normalized

        normalized = trim(adjustl(name))
        select case (normalized)
        case ("none", "numeric", "off")
            code = XGB_CATEGORICAL_NONE
        case ("ordered", "ordered-gradient", "gradient")
            code = XGB_CATEGORICAL_ORDERED
        case default
            code = -1
        end select
    end function parse_categorical_policy

    logical function has_duplicate_sorted_index(values) result(duplicate)
        integer, intent(in) :: values(:)
        integer :: i

        duplicate = .false.
        do i = 2, size(values)
            if (values(i) <= values(i - 1)) then
                duplicate = .true.
                return
            end if
        end do
    end function has_duplicate_sorted_index

    logical function is_integer_code(value) result(valid)
        real(dp), intent(in) :: value

        valid = ieee_is_finite(value)
        if (.not. valid) return
        if (abs(value) > real(huge(0), dp)) then
            valid = .false.
            return
        end if
        valid = value == real(nint(value), dp)
    end function is_integer_code

    logical function valid_categorical_values(x, features, max_categories) result(valid)
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: features(:), max_categories
        integer :: j, i, n_categories, category
        integer, allocatable :: values(:)

        valid = max_categories >= 2 .and. max_categories <= XGB_MAX_CATEGORICAL_VALUES
        if (.not. valid) return
        allocate(values(max_categories))
        do j = 1, size(features)
            if (features(j) < 1 .or. features(j) > size(x, 2)) then
                valid = .false.
                return
            end if
            n_categories = 0
            do i = 1, size(x, 1)
                if (ieee_is_nan(x(i, features(j)))) cycle
                if (.not. is_integer_code(x(i, features(j)))) then
                    valid = .false.
                    return
                end if
                category = nint(x(i, features(j)))
                if (find_category(values, n_categories, category) == 0) then
                    n_categories = n_categories + 1
                    if (n_categories > max_categories) then
                        valid = .false.
                        return
                    end if
                    values(n_categories) = category
                end if
            end do
        end do
    end function valid_categorical_values

    logical function is_categorical_feature(options, feature) result(value)
        type(xgboost_options_t), intent(in) :: options
        integer, intent(in) :: feature

        value = .false.
        if (parse_categorical_policy(options%categorical_policy) /= XGB_CATEGORICAL_ORDERED) return
        if (.not. allocated(options%categorical_features)) return
        value = any(options%categorical_features == feature)
    end function is_categorical_feature

    integer function find_category(values, n_values, category) result(index)
        integer, intent(in) :: values(:), n_values, category
        integer :: i

        index = 0
        do i = 1, n_values
            if (values(i) == category) then
                index = i
                return
            end if
        end do
    end function find_category

    subroutine sort_category_order(values, scores, order, n_values)
        integer, intent(in) :: values(:), n_values
        real(dp), intent(in) :: scores(:)
        integer, intent(inout) :: order(:)
        integer :: i, j, key

        do i = 2, n_values
            key = order(i)
            j = i - 1
            do while (j >= 1)
                if (.not. (scores(order(j)) > scores(key) .or. &
                    (scores(order(j)) == scores(key) .and. values(order(j)) > values(key)))) exit
                order(j + 1) = order(j)
                j = j - 1
            end do
            order(j + 1) = key
        end do
    end subroutine sort_category_order

    integer function missing_code_for_options(options) result(code)
        type(xgboost_options_t), intent(in) :: options

        code = parse_missing_policy(options%missing_policy)
    end function missing_code_for_options

    integer function monotone_constraint_for_feature(options, feature) result(value)
        type(xgboost_options_t), intent(in) :: options
        integer, intent(in) :: feature

        value = 0
        if (.not. allocated(options%monotone_constraints)) return
        if (feature < 1 .or. feature > size(options%monotone_constraints)) return
        value = options%monotone_constraints(feature)
    end function monotone_constraint_for_feature

    integer function interaction_group_for_feature(options, feature) result(value)
        !! Return zero for an unconstrained feature or absent option vector.
        type(xgboost_options_t), intent(in) :: options
        integer, intent(in) :: feature

        value = 0
        if (.not. allocated(options%interaction_groups)) return
        if (feature < 1 .or. feature > size(options%interaction_groups)) return
        value = options%interaction_groups(feature)
    end function interaction_group_for_feature

    logical function valid_query_values(missing_code, x) result(valid)
        integer, intent(in) :: missing_code
        real(dp), intent(in) :: x(:, :)

        valid = .not. any((.not. ieee_is_finite(x)) .and. (.not. ieee_is_nan(x)))
        if (.not. valid) return
        valid = missing_code /= XGB_MISSING_ERROR .or. .not. any(ieee_is_nan(x))
    end function valid_query_values

    logical function valid_categorical_query(model, x) result(valid)
        class(xgboost_t), intent(in) :: model
        real(dp), intent(in) :: x(:, :)
        integer :: j, i, feature

        valid = .true.
        if (model%categorical_policy_code /= XGB_CATEGORICAL_ORDERED) return
        if (.not. allocated(model%categorical_features)) then
            valid = .false.
            return
        end if
        do j = 1, size(model%categorical_features)
            feature = model%categorical_features(j)
            if (feature < 1 .or. feature > size(x, 2)) then
                valid = .false.
                return
            end if
            do i = 1, size(x, 1)
                if (ieee_is_nan(x(i, feature))) cycle
                if (.not. is_integer_code(x(i, feature))) then
                    valid = .false.
                    return
                end if
            end do
        end do
    end function valid_categorical_query

    !> Select deterministic weighted-quantile boundaries for a histogram node.
    !>
    !> `values` and `order` describe the finite values in ascending order;
    !> `sample_index` maps them back to the original rows.  A NaN is never
    !> passed here: the caller keeps it in an explicit missing bin and applies
    !> the configured default direction separately.  Quantile targets are
    !> cumulative observation mass, so sample weights affect bin boundaries but
    !> cannot change their deterministic feature/value tie ordering.
    subroutine histogram_cut_positions(values, order, sample_index, &
            observation_weight, max_bin, positions, n_positions)
        real(dp), intent(in) :: values(:), observation_weight(:)
        integer, intent(in) :: order(:), sample_index(:), max_bin
        integer, intent(out) :: positions(:), n_positions
        integer :: n, n_bins, b, k, position
        real(dp) :: total_weight, cumulative, target

        n = size(values)
        n_positions = 0
        if (n < 2 .or. size(order) /= n .or. size(sample_index) /= n .or. &
            size(positions) < n - 1 .or. max_bin < 2) return
        total_weight = 0.0_dp
        do k = 1, n
            total_weight = total_weight + &
                observation_weight(sample_index(order(k)))
        end do
        if (.not. ieee_is_finite(total_weight) .or. total_weight <= 0.0_dp) return
        n_bins = min(max_bin, n)
        do b = 1, n_bins - 1
            target = total_weight*real(b, dp)/real(n_bins, dp)
            cumulative = 0.0_dp
            position = n
            do k = 1, n - 1
                cumulative = cumulative + &
                    observation_weight(sample_index(order(k)))
                if (cumulative >= target) then
                    position = k
                    exit
                end if
            end do
            if (position >= n) cycle
            if (values(order(position)) >= values(order(position + 1))) cycle
            if (n_positions == 0) then
                n_positions = n_positions + 1
                positions(n_positions) = position
            else if (positions(n_positions) /= position) then
                n_positions = n_positions + 1
                positions(n_positions) = position
            end if
        end do
    end subroutine histogram_cut_positions

    !> Public internal histogram primitive shared by separately named growth
    !> policies.  The XGBoost estimator keeps its historical implementation
    !> private; this narrow wrapper lets LightGBM-style leaf-wise growth use
    !> exactly the same deterministic weighted-quantile cut policy.
    subroutine xgb_histogram_cut_positions(values, order, sample_index, &
            observation_weight, max_bin, positions, n_positions)
        real(dp), intent(in) :: values(:), observation_weight(:)
        integer, intent(in) :: order(:), sample_index(:), max_bin
        integer, intent(out) :: positions(:), n_positions

        call histogram_cut_positions(values, order, sample_index, observation_weight, &
            max_bin, positions, n_positions)
    end subroutine xgb_histogram_cut_positions

    logical function go_left(value, threshold, missing_left) result(value_is_left)
        real(dp), intent(in) :: value, threshold
        logical, intent(in) :: missing_left

        if (ieee_is_nan(value)) then
            value_is_left = missing_left
        else
            value_is_left = value < threshold
        end if
    end function go_left

    logical function category_go_left(value, categories, n_categories, missing_left) result(value_is_left)
        real(dp), intent(in) :: value
        integer, intent(in) :: categories(:), n_categories
        logical, intent(in) :: missing_left
        integer :: i, category

        if (ieee_is_nan(value)) then
            value_is_left = missing_left
            return
        end if
        if (.not. is_integer_code(value)) then
            value_is_left = .false.
            return
        end if
        category = nint(value)
        value_is_left = .false.
        do i = 1, n_categories
            if (categories(i) == category) then
                value_is_left = .true.
                return
            end if
        end do
    end function category_go_left

    real(dp) function stable_logit(probability) result(value)
        real(dp), intent(in) :: probability
        real(dp), parameter :: epsilon = 1.0e-12_dp
        real(dp) :: clipped

        clipped = min(max(probability, epsilon), 1.0_dp - epsilon)
        value = log(clipped) - log(1.0_dp - clipped)
    end function stable_logit

    real(dp) function stable_log_rate(mean_value) result(value)
        !! Finite log-link intercept for a weighted Poisson target mean.
        real(dp), intent(in) :: mean_value
        real(dp), parameter :: minimum_mean = 1.0e-12_dp
        real(dp) :: clipped

        clipped = max(mean_value, minimum_mean)
        value = log(clipped)
        value = min(max(value, log(tiny(1.0_dp))), &
            log(huge(1.0_dp)) - 1.0_dp)
    end function stable_log_rate

    pure real(dp) function stable_poisson_mean(value) result(mean_value)
        !! Overflow/underflow-safe `exp(value)` for Poisson means.
        real(dp), intent(in) :: value
        real(dp) :: clipped

        clipped = min(max(value, log(tiny(1.0_dp))), &
            log(huge(1.0_dp)) - 1.0_dp)
        mean_value = exp(clipped)
    end function stable_poisson_mean

    pure real(dp) function stable_tweedie_exp(margin, exponent) result(value)
        !! Evaluate `exp(exponent*margin)` without allowing an intermediate
        !! product overflow.  The caller validates finite inputs and the
        !! supported Tweedie power interval, so clipping only protects the
        !! tails of otherwise valid objective evaluations.
        real(dp), intent(in) :: margin, exponent
        real(dp), parameter :: lower = log(tiny(1.0_dp))
        real(dp), parameter :: upper = log(huge(1.0_dp)) - 1.0_dp
        real(dp) :: scaled

        scaled = exponent*margin
        if (.not. ieee_is_finite(scaled)) then
            if ((exponent > 0.0_dp .and. margin > 0.0_dp) .or. &
                (exponent < 0.0_dp .and. margin < 0.0_dp)) then
                scaled = upper
            else
                scaled = lower
            end if
        else
            scaled = min(max(scaled, lower), upper)
        end if
        value = exp(scaled)
    end function stable_tweedie_exp

    pure elemental real(dp) function stable_poisson_element(value) result(mean_value)
        real(dp), intent(in) :: value

        mean_value = stable_poisson_mean(value)
    end function stable_poisson_element

    function stable_poisson_array(values) result(means)
        real(dp), intent(in) :: values(:)
        real(dp) :: means(size(values))

        means = stable_poisson_element(values)
    end function stable_poisson_array

    pure real(dp) function stable_squared_log_mean(value) result(mean_value)
        !! Overflow/underflow-safe inverse link `expm1(value)`.
        !!
        !! Tree updates can temporarily move a margin below zero even though
        !! the final prediction is constrained to be at least -1 by the
        !! squared-log link.  Clipping before exponentiation keeps both tails
        !! finite without changing ordinary double-precision values.
        real(dp), intent(in) :: value
        real(dp) :: clipped

        clipped = min(max(value, log(tiny(1.0_dp))), &
            log(huge(1.0_dp)) - 1.0_dp)
        mean_value = exp(clipped) - 1.0_dp
    end function stable_squared_log_mean

    pure elemental real(dp) function stable_squared_log_element(value) &
            result(mean_value)
        real(dp), intent(in) :: value

        mean_value = stable_squared_log_mean(value)
    end function stable_squared_log_element

    function stable_squared_log_array(values) result(means)
        real(dp), intent(in) :: values(:)
        real(dp) :: means(size(values))

        means = stable_squared_log_element(values)
    end function stable_squared_log_array

    pure real(dp) function safe_midpoint(left, right) result(midpoint)
        !! Overflow-safe midpoint for finite ordered values.  Computing each
        !! half before adding keeps valid same-sign values near `huge()` finite.
        real(dp), intent(in) :: left, right

        midpoint = 0.5_dp*left + 0.5_dp*right
    end function safe_midpoint

    real(dp) function stable_sigmoid(value) result(probability)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            probability = exp(value)/(1.0_dp + exp(value))
        end if
    end function stable_sigmoid

    pure elemental real(dp) function stable_sigmoid_element(value) result(probability)
        real(dp), intent(in) :: value

        if (value >= 0.0_dp) then
            probability = 1.0_dp/(1.0_dp + exp(-value))
        else
            probability = exp(value)/(1.0_dp + exp(value))
        end if
    end function stable_sigmoid_element

    function stable_sigmoid_array(values) result(probabilities)
        real(dp), intent(in) :: values(:)
        real(dp) :: probabilities(size(values))

        probabilities = stable_sigmoid_element(values)
    end function stable_sigmoid_array

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

    real(dp) function weighted_quantile(values, weights, alpha) result(quantile)
        real(dp), intent(in) :: values(:), weights(:), alpha
        real(dp), allocatable :: sorted_values(:), sorted_weights(:)
        real(dp) :: value_key, weight_key, threshold, cumulative
        integer :: i, j, n

        n = size(values)
        allocate(sorted_values(n), sorted_weights(n))
        sorted_values = values
        sorted_weights = weights
        do i = 2, n
            value_key = sorted_values(i)
            weight_key = sorted_weights(i)
            j = i - 1
            do while (j >= 1)
                if (sorted_values(j) <= value_key) exit
                sorted_values(j + 1) = sorted_values(j)
                sorted_weights(j + 1) = sorted_weights(j)
                j = j - 1
            end do
            sorted_values(j + 1) = value_key
            sorted_weights(j + 1) = weight_key
        end do
        threshold = alpha*sum(sorted_weights)
        cumulative = 0.0_dp
        quantile = sorted_values(n)
        do i = 1, n
            cumulative = cumulative + sorted_weights(i)
            if (cumulative >= threshold) then
                quantile = sorted_values(i)
                exit
            end if
        end do
    end function weighted_quantile

end module fortml_xgboost
