!> One-vs-rest multiclass classification over the exact XGBoost-style binary
!> logistic estimator.
module fortml_xgboost_multiclass
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan
    use, intrinsic :: iso_fortran_env, only: iostat_end
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, &
        FORTML_DEVICE_CUDA
    use fortml_xgboost, only: xgboost_t, xgboost_options_t, XGB_MODEL_TEXT_MAGIC
    implicit none
    private

    character(*), parameter, public :: XGB_MULTICLASS_MODEL_TEXT_MAGIC = &
        "FORTML_XGBOOST_MULTICLASS_TEXT"
    integer, parameter, public :: XGB_MULTICLASS_MODEL_TEXT_SCHEMA_VERSION = 2

    !> Deterministic one-vs-rest multiclass XGBoost-style classifier.
    !>
    !> Each class owns a deterministic depth-limited second-order logistic
    !> booster.  Positive class probabilities are normalized across the
    !> one-vs-rest models, so
    !> arbitrary integer labels and a proper multiclass probability contract
    !> are available without changing the binary estimator.
    type, public :: xgboost_multiclass_t
        private
        type(xgboost_t), allocatable :: one_vs_rest(:)
        integer, allocatable :: class_label(:)
        integer :: n_inputs = 0
        integer :: requested_estimators = 0
        integer :: best_iteration_value = 0
        real(dp) :: best_validation_loss_value = huge(1.0_dp)
        logical :: early_stopped_flag = .false.
        logical :: initialized = .false.
    contains
        procedure, public :: fit => xgb_multiclass_fit
        procedure, public :: save_text => xgb_multiclass_save_text
        procedure, public :: load_text => xgb_multiclass_load_text
        procedure, public :: predict_proba => xgb_multiclass_predict_proba
        procedure, public :: predict_proba_device => &
            xgb_multiclass_predict_proba_device
        procedure, public :: predict_proba_staged => &
            xgb_multiclass_predict_proba_staged
        procedure, public :: decision_function_staged => &
            xgb_multiclass_decision_function_staged
        procedure, public :: predict_proba_jvp => xgb_multiclass_predict_proba_jvp
        procedure, public :: predict_proba_vjp => xgb_multiclass_predict_proba_vjp
        procedure, public :: decision_function => xgb_multiclass_decision_function
        procedure, public :: predict => xgb_multiclass_predict
        procedure, public :: predict_device => xgb_multiclass_predict_device
        procedure, public :: device_supported => xgb_multiclass_device_supported
        procedure, public :: feature_importance => &
            xgb_multiclass_feature_importance
        procedure, public :: classes => xgb_multiclass_classes
        procedure, public :: feature_count => xgb_multiclass_feature_count
        procedure, public :: class_count => xgb_multiclass_class_count
        procedure, public :: estimator_count => xgb_multiclass_estimator_count
        procedure, public :: requested_estimator_count => &
            xgb_multiclass_requested_estimator_count
        procedure, public :: best_iteration => xgb_multiclass_best_iteration
        procedure, public :: best_validation_loss => &
            xgb_multiclass_best_validation_loss
        procedure, public :: early_stopped => xgb_multiclass_early_stopped
        procedure, public :: monotone_constraint => &
            xgb_multiclass_monotone_constraint
        procedure, public :: fitted => xgb_multiclass_fitted
    end type xgboost_multiclass_t

contains

    subroutine xgb_multiclass_fit(self, x, labels, status, options, sample_weight, &
            validation_x, validation_labels, validation_weight)
        class(xgboost_multiclass_t), intent(inout) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(in) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_options_t), intent(in), optional :: options
        real(dp), intent(in), optional :: sample_weight(:)
        real(dp), intent(in), optional :: validation_x(:, :)
        integer, intent(in), optional :: validation_labels(:)
        real(dp), intent(in), optional :: validation_weight(:)
        type(xgboost_options_t) :: settings, child_settings
        type(xgboost_multiclass_t) :: candidate
        type(xgboost_t) :: sliced
        integer, allocatable :: classes(:), binary_labels(:), validation_binary(:)
        real(dp), allocatable :: staged(:, :, :), child_staged(:, :, :), totals(:, :)
        real(dp), allocatable :: validation_observation_weight(:)
        real(dp) :: validation_loss, best_validation_loss, weight_sum
        integer :: i, j, n_classes, n_samples, n_validation, completed_estimators
        integer :: best_iteration, stale_rounds, keep_estimators
        logical :: have_validation, improved, known_label
        !! Default-initialized instances, standing in for empty
        !! structure constructors: nvfortran rejects `T()` outright,
        !! and a declared local carries the same default init.
        type(xgboost_options_t) :: xgboost_options_t_default

        settings = xgboost_options_t_default
        if (present(options)) settings = options
        have_validation = present(validation_x) .or. present(validation_labels) .or. &
            present(validation_weight)
        if (present(validation_x) .neqv. present(validation_labels)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass fit: validation_x and validation_labels must be supplied together")
            return
        end if
        if (present(validation_weight) .and. .not. present(validation_x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass fit: validation_weight requires validation data")
            return
        end if
        if (settings%early_stopping_rounds < 0 .or. &
            .not. ieee_is_finite(settings%early_stopping_min_delta) .or. &
            settings%early_stopping_min_delta < 0.0_dp .or. &
            (settings%early_stopping_rounds > 0 .and. .not. have_validation)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass fit: invalid early-stopping configuration")
            return
        end if
        n_samples = size(x, 1)
        if (n_samples < 2 .or. size(x, 2) < 1 .or. size(labels) /= n_samples) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass fit: input dimensions are invalid")
            return
        end if
        if (present(sample_weight)) then
            if (size(sample_weight) /= n_samples .or. &
                any(.not. ieee_is_finite(sample_weight)) .or. &
                any(sample_weight <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "XGBoost multiclass fit: sample_weight must be positive and finite")
                return
            end if
        end if
        call unique_sorted_labels(labels, classes)
        n_classes = size(classes)
        if (n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass fit: at least two classes are required")
            return
        end if
        if (have_validation) then
            n_validation = size(validation_x, 1)
            if (n_validation < 1 .or. size(validation_x, 2) /= size(x, 2) .or. &
                size(validation_labels) /= n_validation) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "XGBoost multiclass fit: validation dimensions do not match training features")
                return
            end if
            if (present(validation_weight)) then
                if (size(validation_weight) /= n_validation .or. &
                    any(.not. ieee_is_finite(validation_weight)) .or. &
                    any(validation_weight <= 0.0_dp)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "XGBoost multiclass fit: validation_weight must be positive and finite")
                    return
                end if
            end if
            do i = 1, n_validation
                known_label = any(validation_labels(i) == classes)
                if (.not. known_label) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "XGBoost multiclass fit: validation labels must belong to training classes")
                    return
                end if
            end do
        else
            n_validation = 0
        end if

        ! Every child must retain the same complete tree prefix.  Validation
        ! stopping is therefore evaluated once on the normalized multiclass
        ! probabilities below; the binary estimator still performs all of its
        ! usual validation/input checks and staged prediction work.
        child_settings = settings
        child_settings%early_stopping_rounds = 0
        child_settings%restore_best = .false.
        allocate(candidate%one_vs_rest(n_classes), candidate%class_label(n_classes))
        candidate%class_label = classes
        allocate(binary_labels(n_samples))
        if (have_validation) allocate(validation_binary(n_validation))
        do i = 1, n_classes
            binary_labels = 0
            where (labels == classes(i)) binary_labels = 1
            if (have_validation) then
                validation_binary = 0
                where (validation_labels == classes(i)) validation_binary = 1
            end if
            if (present(sample_weight)) then
                if (have_validation) then
                    if (present(validation_weight)) then
                        call candidate%one_vs_rest(i)%fit_binary(x, real(binary_labels, dp), &
                            status, child_settings, sample_weight, validation_x, &
                            real(validation_binary, dp), validation_weight)
                    else
                        call candidate%one_vs_rest(i)%fit_binary(x, real(binary_labels, dp), &
                            status, child_settings, sample_weight, validation_x, &
                            real(validation_binary, dp))
                    end if
                else
                    call candidate%one_vs_rest(i)%fit_binary(x, real(binary_labels, dp), &
                        status, child_settings, sample_weight)
                end if
            else if (have_validation) then
                if (present(validation_weight)) then
                    call candidate%one_vs_rest(i)%fit_binary(x, real(binary_labels, dp), &
                        status, child_settings, validation_x=validation_x, &
                        validation_y=real(validation_binary, dp), &
                        validation_weight=validation_weight)
                else
                    call candidate%one_vs_rest(i)%fit_binary(x, real(binary_labels, dp), &
                        status, child_settings, validation_x=validation_x, &
                        validation_y=real(validation_binary, dp))
                end if
            else
                call candidate%one_vs_rest(i)%fit_binary(x, real(binary_labels, dp), &
                    status, child_settings)
            end if
            if (status%code /= FORTNUM_OK) return
        end do
        candidate%n_inputs = size(x, 2)
        candidate%requested_estimators = settings%n_estimators
        candidate%best_iteration_value = settings%n_estimators
        candidate%best_validation_loss_value = huge(1.0_dp)
        candidate%early_stopped_flag = .false.
        completed_estimators = candidate%one_vs_rest(1)%estimator_count()

        if (have_validation) then
            allocate(staged(n_validation, n_classes, completed_estimators), &
                child_staged(n_validation, 2, completed_estimators), &
                totals(n_validation, completed_estimators), &
                validation_observation_weight(n_validation))
            validation_observation_weight = 1.0_dp
            if (present(validation_weight)) validation_observation_weight = validation_weight
            weight_sum = sum(validation_observation_weight)
            if (.not. ieee_is_finite(weight_sum) .or. weight_sum <= 0.0_dp) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "XGBoost multiclass fit: validation_weight has no positive mass")
                return
            end if
            staged = 0.0_dp
            do i = 1, n_classes
                call candidate%one_vs_rest(i)%predict_proba_staged(validation_x, &
                    child_staged, status)
                if (status%code /= FORTNUM_OK) return
                staged(:, i, :) = child_staged(:, 2, :)
            end do
            totals = sum(staged, dim=2)
            if (any(.not. ieee_is_finite(totals)) .or. any(totals <= 0.0_dp)) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "XGBoost multiclass fit: validation normalization failed")
                return
            end if
            best_validation_loss = huge(1.0_dp)
            best_iteration = 0
            stale_rounds = 0
            do j = 1, completed_estimators
                validation_loss = multiclass_log_loss(staged(:, :, j), &
                    totals(:, j), validation_labels, classes, validation_observation_weight, &
                    weight_sum)
                if (.not. ieee_is_finite(validation_loss)) then
                    call status_set(status, FORTNUM_DOMAIN_ERROR, &
                        "XGBoost multiclass fit: validation objective is nonfinite")
                    return
                end if
                improved = validation_loss < best_validation_loss - &
                    settings%early_stopping_min_delta
                if (improved) then
                    best_validation_loss = validation_loss
                    best_iteration = j
                    stale_rounds = 0
                else
                    stale_rounds = stale_rounds + 1
                end if
                if (settings%early_stopping_rounds > 0 .and. &
                    stale_rounds >= settings%early_stopping_rounds) then
                    candidate%early_stopped_flag = .true.
                    exit
                end if
            end do
            if (best_iteration < 1) then
                call status_set(status, FORTNUM_DOMAIN_ERROR, &
                    "XGBoost multiclass fit: validation objective did not produce a finite score")
                return
            end if
            candidate%best_iteration_value = best_iteration
            candidate%best_validation_loss_value = best_validation_loss
            keep_estimators = completed_estimators
            if (settings%restore_best) keep_estimators = best_iteration
            if (candidate%early_stopped_flag .and. .not. settings%restore_best) then
                keep_estimators = j
            end if
            if (keep_estimators < 1) keep_estimators = best_iteration
            if (keep_estimators < completed_estimators) then
                do i = 1, n_classes
                    call candidate%one_vs_rest(i)%slice(keep_estimators, sliced, status)
                    if (status%code /= FORTNUM_OK) return
                    candidate%one_vs_rest(i) = sliced
                end do
            end if
        end if
        candidate%n_inputs = size(x, 2)
        candidate%initialized = .true.
        if (.not. valid_multiclass_model(candidate)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass fit: child model metadata is inconsistent")
            return
        end if
        self%one_vs_rest = candidate%one_vs_rest
        self%class_label = candidate%class_label
        self%n_inputs = candidate%n_inputs
        self%requested_estimators = candidate%requested_estimators
        self%best_iteration_value = candidate%best_iteration_value
        self%best_validation_loss_value = candidate%best_validation_loss_value
        self%early_stopped_flag = candidate%early_stopped_flag
        self%initialized = candidate%initialized
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_fit

    subroutine xgb_multiclass_save_text(self, path, status)
        !! Save the complete OVR ensemble as one portable text snapshot.
        !!
        !! Each private binary ensemble is written with the established
        !! XGBoost text serializer and copied into a child section.  The
        !! resulting artifact has no sidecar files and can be moved between
        !! compiler builds without exposing the child tree representation.
        class(xgboost_multiclass_t), intent(in) :: self
        character(*), intent(in) :: path
        type(fortnum_status_t), intent(out) :: status
        integer :: unit, child_unit, ios, child_ios, close_ios, i
        logical :: child_open
        character(len=1024) :: child_path
        character(len=1024) :: line

        if (.not. valid_multiclass_model(self)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass save_text: model is not a valid fitted ensemble")
            return
        end if
        if (len_trim(path) < 1 .or. len_trim(path) > 900) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass save_text: destination path is invalid")
            return
        end if
        open(newunit=unit, file=path, status="replace", action="write", &
            form="formatted", access="sequential", iostat=ios)
        if (ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass save_text: cannot open destination")
            return
        end if
        write(unit, "(A)", iostat=ios) XGB_MULTICLASS_MODEL_TEXT_MAGIC
        if (ios == 0) call multi_write_i(unit, "schema_version", &
            XGB_MULTICLASS_MODEL_TEXT_SCHEMA_VERSION, ios)
        if (ios == 0) call multi_write_i(unit, "n_inputs", self%n_inputs, ios)
        if (ios == 0) call multi_write_i(unit, "class_count", self%class_count(), ios)
        if (ios == 0) call multi_write_i(unit, "requested_estimators", &
            self%requested_estimators, ios)
        if (ios == 0) call multi_write_i(unit, "best_iteration", &
            self%best_iteration_value, ios)
        if (ios == 0) call multi_write_r(unit, "best_validation_loss", &
            self%best_validation_loss_value, ios)
        if (ios == 0) call multi_write_l(unit, "early_stopped", &
            self%early_stopped_flag, ios)
        do i = 1, self%class_count()
            if (ios /= 0) exit
            call multi_write_i(unit, "class_label", self%class_label(i), ios)
        end do
        child_open = .false.
        do i = 1, self%class_count()
            if (ios /= 0) exit
            call multi_write_i(unit, "child_begin", i, ios)
            call make_child_path(path, i, child_path, ios)
            if (ios /= 0) exit
            call self%one_vs_rest(i)%save_text(trim(child_path), status)
            if (status%code /= FORTNUM_OK) then
                ios = 1
                exit
            end if
            open(newunit=child_unit, file=trim(child_path), status="old", &
                action="read", form="formatted", access="sequential", &
                iostat=child_ios)
            if (child_ios /= 0) then
                ios = 1
                call delete_file(trim(child_path))
                exit
            end if
            child_open = .true.
            do
                read(child_unit, "(A)", iostat=child_ios) line
                if (child_ios == iostat_end) exit
                if (child_ios /= 0) then
                    ios = 1
                    exit
                end if
                write(unit, "(A)", iostat=ios) trim(line)
                if (ios /= 0) exit
            end do
            close_ios = 0
            close(child_unit, iostat=close_ios)
            child_open = .false.
            call delete_file(trim(child_path))
            if (ios == 0 .and. close_ios == 0) then
                call multi_write_i(unit, "child_end", i, ios)
            else
                ios = 1
            end if
        end do
        if (ios == 0) write(unit, "(A)", iostat=ios) "end"
        if (child_open) close(child_unit, iostat=close_ios)
        close_ios = 0
        close(unit, iostat=close_ios)
        if (ios /= 0 .or. close_ios /= 0) then
            call delete_file(trim(child_path))
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass save_text: formatted write failed")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_save_text

    subroutine xgb_multiclass_load_text(self, path, status)
        !! Load a one-file OVR snapshot without mutating the destination on
        !! malformed, truncated, or incompatible input.
        class(xgboost_multiclass_t), intent(inout) :: self
        character(*), intent(in) :: path
        type(fortnum_status_t), intent(out) :: status
        type(xgboost_multiclass_t) :: candidate
        integer :: unit, child_unit, ios, close_ios, schema, n_inputs
        integer :: class_count, i, child_index
        integer :: requested_estimators, best_iteration
        real(dp) :: best_validation_loss
        logical :: early_stopped
        logical :: child_open
        character(len=1024) :: line, key
        character(len=1024) :: child_path

        child_open = .false.
        child_path = ""
        open(newunit=unit, file=path, status="old", action="read", &
            form="formatted", access="sequential", iostat=ios)
        if (ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass load_text: cannot open source")
            return
        end if
        read(unit, "(A)", iostat=ios) line
        if (ios /= 0 .or. trim(line) /= XGB_MULTICLASS_MODEL_TEXT_MAGIC) goto 900
        call multi_read_i(unit, "schema_version", schema, ios)
        if (ios /= 0 .or. schema /= XGB_MULTICLASS_MODEL_TEXT_SCHEMA_VERSION) goto 900
        call multi_read_i(unit, "n_inputs", n_inputs, ios)
        if (ios /= 0 .or. n_inputs < 1) goto 900
        call multi_read_i(unit, "class_count", class_count, ios)
        if (ios /= 0 .or. class_count < 2) goto 900
        call multi_read_i(unit, "requested_estimators", requested_estimators, ios)
        if (ios /= 0 .or. requested_estimators < 1) goto 900
        call multi_read_i(unit, "best_iteration", best_iteration, ios)
        call multi_read_r(unit, "best_validation_loss", best_validation_loss, ios)
        call multi_read_l(unit, "early_stopped", early_stopped, ios)
        if (ios /= 0 .or. best_iteration < 1 .or. &
            .not. ieee_is_finite(best_validation_loss)) goto 900
        allocate(candidate%class_label(class_count), candidate%one_vs_rest(class_count), &
            stat=ios)
        if (ios /= 0) goto 900
        candidate%n_inputs = n_inputs
        candidate%requested_estimators = requested_estimators
        candidate%best_iteration_value = best_iteration
        candidate%best_validation_loss_value = best_validation_loss
        candidate%early_stopped_flag = early_stopped
        do i = 1, class_count
            call multi_read_i(unit, "class_label", candidate%class_label(i), ios)
            if (ios /= 0) goto 900
            if (i > 1) then
                if (candidate%class_label(i) <= candidate%class_label(i - 1)) goto 900
            end if
        end do
        do i = 1, class_count
            read(unit, "(A)", iostat=ios) line
            if (ios /= 0) goto 900
            read(line, *, iostat=ios) key, child_index
            if (ios /= 0 .or. trim(key) /= "child_begin" .or. child_index /= i) goto 900
            call make_child_path(path, i, child_path, ios)
            if (ios /= 0) goto 900
            open(newunit=child_unit, file=trim(child_path), status="replace", &
                action="write", form="formatted", access="sequential", &
                iostat=ios)
            if (ios /= 0) goto 900
            child_open = .true.
            read(unit, "(A)", iostat=ios) line
            if (ios /= 0 .or. trim(line) /= XGB_MODEL_TEXT_MAGIC) goto 900
            write(child_unit, "(A)", iostat=ios) trim(line)
            if (ios /= 0) goto 900
            do
                read(unit, "(A)", iostat=ios) line
                if (ios /= 0) goto 900
                write(child_unit, "(A)", iostat=ios) trim(line)
                if (ios /= 0) goto 900
                if (trim(line) == "end") exit
            end do
            close_ios = 0
            close(child_unit, iostat=close_ios)
            child_open = .false.
            if (close_ios /= 0) goto 900
            read(unit, "(A)", iostat=ios) line
            if (ios /= 0) goto 900
            read(line, *, iostat=ios) key, child_index
            if (ios /= 0 .or. trim(key) /= "child_end" .or. child_index /= i) goto 900
            call candidate%one_vs_rest(i)%load_text(trim(child_path), status)
            call delete_file(trim(child_path))
            if (status%code /= FORTNUM_OK) then
                ios = 1
                goto 900
            end if
            if (candidate%one_vs_rest(i)%feature_count() /= n_inputs) goto 900
            if (trim(candidate%one_vs_rest(i)%objective_name()) /= "logistic") goto 900
        end do
        read(unit, "(A)", iostat=ios) line
        if (ios /= 0 .or. trim(line) /= "end") goto 900
        read(unit, "(A)", iostat=ios) line
        if (ios /= iostat_end) goto 900
        close_ios = 0
        close(unit, iostat=close_ios)
        if (close_ios /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass load_text: malformed snapshot")
            return
        end if
        candidate%initialized = .true.
        if (.not. valid_multiclass_model(candidate)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass load_text: child metadata is inconsistent")
            return
        end if
        select type(destination => self)
            type is (xgboost_multiclass_t)
            destination = candidate
        class default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass load_text: destination type is unsupported")
            return
        end select
        call status_set(status, FORTNUM_OK, "")
        return

        900     continue
        if (child_open) close(child_unit, iostat=close_ios)
        call delete_file(trim(child_path))
        close_ios = 0
        close(unit, iostat=close_ios)
        call status_set(status, FORTNUM_DOMAIN_ERROR, &
            "XGBoost multiclass load_text: malformed, truncated, or unsupported snapshot")
    end subroutine xgb_multiclass_load_text

    subroutine xgb_multiclass_predict_proba(self, x, probabilities, status)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_probabilities(:, :), totals(:)
        integer :: i

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass predict_proba: model or input is invalid")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), self%class_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass predict_proba: output shape is invalid")
            return
        end if
        allocate(binary_probabilities(size(x, 1), 2), totals(size(x, 1)))
        probabilities = 0.0_dp
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_proba(x, binary_probabilities, &
                status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, i) = binary_probabilities(:, 2)
        end do
        totals = sum(probabilities, dim=2)
        if (any(.not. ieee_is_finite(totals)) .or. any(totals <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass predict_proba: normalization failed")
            return
        end if
        do i = 1, self%class_count()
            probabilities(:, i) = probabilities(:, i)/totals
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_predict_proba

    subroutine xgb_multiclass_predict_proba_device(self, device, x, probabilities, &
            status)
        !! Predict normalized OVR probabilities through the explicit device
        !! contract.  CUDA remains a typed refusal until a resident
        !! multiclass/tree kernel is linked; host fallback is never hidden.
        class(xgboost_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict_proba(x, probabilities, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "XGBoost multiclass device prediction: no resident CUDA tree kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass device prediction: device kind is invalid")
        end select
    end subroutine xgb_multiclass_predict_proba_device

    !> Return normalized one-vs-rest probabilities after every boosting stage.
    !>
    !> The output has shape `(n_samples, n_classes, n_estimators)`.  Each
    !> stage is normalized across classes, just like `predict_proba`.
    subroutine xgb_multiclass_predict_proba_staged(self, x, probabilities, &
            status)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: probabilities(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_staged(:, :), totals(:, :)
        integer :: i

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass predict_proba_staged: model or input is invalid")
            return
        end if
        if (size(probabilities, 1) /= size(x, 1) .or. &
            size(probabilities, 2) /= self%class_count() .or. &
            size(probabilities, 3) /= self%estimator_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass predict_proba_staged: output shape is invalid")
            return
        end if
        allocate(binary_staged(size(x, 1), self%estimator_count()), &
            totals(size(x, 1), self%estimator_count()))
        probabilities = 0.0_dp
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_staged(x, binary_staged, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, i, :) = binary_staged
        end do
        totals = sum(probabilities, dim=2)
        if (any(.not. ieee_is_finite(totals)) .or. any(totals <= 0.0_dp)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass predict_proba_staged: normalization failed")
            return
        end if
        do i = 1, self%class_count()
            probabilities(:, i, :) = probabilities(:, i, :)/totals
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_predict_proba_staged

    !> Return one-vs-rest raw margins after every boosting stage.
    !>
    !> The output has shape `(n_samples, n_classes, n_estimators)` and is not
    !> normalized; use `predict_proba_staged` for class probabilities.
    subroutine xgb_multiclass_decision_function_staged(self, x, margins, status)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margins(:, :, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: binary_margins(:, :)
        integer :: i

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass decision_function_staged: model or "// &
                "input is invalid")
            return
        end if
        if (size(margins, 1) /= size(x, 1) .or. &
            size(margins, 2) /= self%class_count() .or. &
            size(margins, 3) /= self%estimator_count()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass decision_function_staged: output shape is invalid")
            return
        end if
        allocate(binary_margins(size(x, 1), self%estimator_count()))
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_staged_margin(x, binary_margins, status)
            if (status%code /= FORTNUM_OK) return
            margins(:, i, :) = binary_margins
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_decision_function_staged

    !> Aggregate feature diagnostics across all one-vs-rest estimators.
    !>
    !> The `kind` and normalization semantics match the binary estimator's
    !> `feature_importance` method.  Raw values are summed across classes.
    subroutine xgb_multiclass_feature_importance(self, importance, status, &
            kind, normalize)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(out) :: importance(:)
        type(fortnum_status_t), intent(out) :: status
        character(len=*), intent(in), optional :: kind
        logical, intent(in), optional :: normalize
        real(dp), allocatable :: contribution(:)
        logical :: should_normalize
        real(dp) :: total
        integer :: i

        if (.not. self%initialized .or. .not. allocated(self%one_vs_rest)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass feature_importance: model is not initialized")
            return
        end if
        if (size(importance) /= self%n_inputs) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass feature_importance: output shape is invalid")
            return
        end if
        allocate(contribution(self%n_inputs))
        importance = 0.0_dp
        do i = 1, self%class_count()
            if (present(kind)) then
                call self%one_vs_rest(i)%feature_importance(contribution, status, &
                    kind, .false.)
            else
                call self%one_vs_rest(i)%feature_importance(contribution, status)
            end if
            if (status%code /= FORTNUM_OK) return
            importance = importance + contribution
        end do
        should_normalize = .false.
        if (present(normalize)) should_normalize = normalize
        if (should_normalize) then
            total = sum(importance)
            if (total > 0.0_dp) importance = importance/total
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_feature_importance

    subroutine xgb_multiclass_predict_proba_jvp(self, x, x_dot, probabilities, &
            probabilities_dot, status)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), x_dot(:, :)
        real(dp), intent(out) :: probabilities(:, :), probabilities_dot(:, :)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: positive(:), positive_dot(:), totals(:), &
            totals_dot(:)
        integer :: i

        if (.not. valid_query(self, x) .or. any(shape(x_dot) /= shape(x))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass probability JVP: model or input is invalid")
            return
        end if
        if (any(.not. ieee_is_finite(x_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass probability JVP: tangent is not finite")
            return
        end if
        if (any(shape(probabilities) /= [size(x, 1), self%class_count()]) .or. &
            any(shape(probabilities_dot) /= shape(probabilities))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass probability JVP: output shape is invalid")
            return
        end if
        allocate(positive(size(x, 1)), positive_dot(size(x, 1)), &
            totals(size(x, 1)), totals_dot(size(x, 1)))
        probabilities = 0.0_dp
        probabilities_dot = 0.0_dp
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_jvp(x, x_dot, positive, &
                positive_dot, status)
            if (status%code /= FORTNUM_OK) return
            probabilities(:, i) = positive
            probabilities_dot(:, i) = positive_dot
        end do
        totals = sum(probabilities, dim=2)
        totals_dot = sum(probabilities_dot, dim=2)
        if (any(.not. ieee_is_finite(totals)) .or. any(totals <= 0.0_dp) .or. &
            any(.not. ieee_is_finite(totals_dot))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass probability JVP: normalization failed")
            return
        end if
        do i = 1, self%class_count()
            probabilities_dot(:, i) = (probabilities_dot(:, i)*totals - &
                probabilities(:, i)*totals_dot)/(totals*totals)
            probabilities(:, i) = probabilities(:, i)/totals
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_predict_proba_jvp

    !> Reverse-mode product of normalized one-vs-rest probabilities with
    !> respect to query features.  It is zero away from tree split surfaces;
    !> the binary estimators provide the shared boundary refusal.
    subroutine xgb_multiclass_predict_proba_vjp(self, x, probabilities_bar, &
            x_bar, status)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :), probabilities_bar(:, :)
        real(dp), intent(out) :: x_bar(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        x_bar = 0.0_dp
        if (.not. valid_query(self, x) .or. &
            any(shape(probabilities_bar) /= [size(x, 1), self%class_count()]) .or. &
            any(shape(x_bar) /= shape(x)) .or. &
            any(.not. ieee_is_finite(probabilities_bar))) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass probability VJP: model, cotangent, or output shape is invalid")
            return
        end if
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%predict_vjp(x, probabilities_bar(:, i), &
                x_bar, status)
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_predict_proba_vjp

    subroutine xgb_multiclass_decision_function(self, x, margins, status)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        real(dp), intent(out) :: margins(:, :)
        type(fortnum_status_t), intent(out) :: status
        integer :: i

        if (.not. valid_query(self, x)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass decision_function: model or input is invalid")
            return
        end if
        if (any(shape(margins) /= [size(x, 1), self%class_count()])) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass decision_function: output shape is invalid")
            return
        end if
        do i = 1, self%class_count()
            call self%one_vs_rest(i)%decision_function(x, margins(:, i), status)
            if (status%code /= FORTNUM_OK) return
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_decision_function

    subroutine xgb_multiclass_predict(self, x, labels, status)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status
        real(dp), allocatable :: probabilities(:, :)
        integer :: i

        if (size(labels) /= size(x, 1)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass predict: output shape is invalid")
            return
        end if
        allocate(probabilities(size(x, 1), self%class_count()))
        call self%predict_proba(x, probabilities, status)
        if (status%code /= FORTNUM_OK) return
        do i = 1, size(labels)
            labels(i) = self%class_label(maxloc(probabilities(i, :), dim=1))
        end do
        call status_set(status, FORTNUM_OK, "")
    end subroutine xgb_multiclass_predict

    subroutine xgb_multiclass_predict_device(self, device, x, labels, status)
        class(xgboost_multiclass_t), intent(in) :: self
        type(fortml_device_t), intent(in) :: device
        real(dp), intent(in) :: x(:, :)
        integer, intent(out) :: labels(:)
        type(fortnum_status_t), intent(out) :: status

        if (.not. device%selected .or. .not. device%available) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass device prediction: device is not selected")
            return
        end if
        select case (device%kind)
        case (FORTML_DEVICE_CPU)
            call self%predict(x, labels, status)
        case (FORTML_DEVICE_CUDA)
            call status_set(status, FORTNUM_NOT_IMPLEMENTED, &
                "XGBoost multiclass device prediction: no resident CUDA tree kernel is linked")
        case default
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "XGBoost multiclass device prediction: device kind is invalid")
        end select
    end subroutine xgb_multiclass_predict_device

    function xgb_multiclass_classes(self) result(classes)
        class(xgboost_multiclass_t), intent(in) :: self
        integer, allocatable :: classes(:)

        if (allocated(self%class_label)) then
            allocate(classes(size(self%class_label)))
            classes = self%class_label
        else
            allocate(classes(0))
        end if
    end function xgb_multiclass_classes

    integer function xgb_multiclass_feature_count(self) result(count)
        class(xgboost_multiclass_t), intent(in) :: self

        count = self%n_inputs
    end function xgb_multiclass_feature_count

    integer function xgb_multiclass_class_count(self) result(count)
        class(xgboost_multiclass_t), intent(in) :: self

        count = 0
        if (allocated(self%class_label)) count = size(self%class_label)
    end function xgb_multiclass_class_count

    integer function xgb_multiclass_estimator_count(self) result(count)
        class(xgboost_multiclass_t), intent(in) :: self

        count = 0
        if (.not. allocated(self%one_vs_rest)) return
        if (size(self%one_vs_rest) < 1) return
        count = self%one_vs_rest(1)%estimator_count()
    end function xgb_multiclass_estimator_count

    integer function xgb_multiclass_requested_estimator_count(self) result(count)
        class(xgboost_multiclass_t), intent(in) :: self

        count = self%requested_estimators
    end function xgb_multiclass_requested_estimator_count

    integer function xgb_multiclass_best_iteration(self) result(iteration)
        !! One-based boosting round with the lowest normalized validation
        !! multiclass log-loss.  Without validation this is the requested
        !! estimator count.
        class(xgboost_multiclass_t), intent(in) :: self

        iteration = self%best_iteration_value
    end function xgb_multiclass_best_iteration

    real(dp) function xgb_multiclass_best_validation_loss(self) result(loss)
        !! Best weighted normalized multiclass validation log-loss.  It is
        !! `huge()` when no validation set was supplied.
        class(xgboost_multiclass_t), intent(in) :: self

        loss = self%best_validation_loss_value
    end function xgb_multiclass_best_validation_loss

    logical function xgb_multiclass_early_stopped(self) result(stopped)
        class(xgboost_multiclass_t), intent(in) :: self

        stopped = self%early_stopped_flag
    end function xgb_multiclass_early_stopped

    integer function xgb_multiclass_monotone_constraint(self, feature_index) &
            result(value)
        class(xgboost_multiclass_t), intent(in) :: self
        integer, intent(in) :: feature_index

        value = 0
        if (.not. self%initialized .or. .not. allocated(self%one_vs_rest)) return
        if (size(self%one_vs_rest) < 1) return
        value = self%one_vs_rest(1)%monotone_constraint(feature_index)
    end function xgb_multiclass_monotone_constraint

    logical function xgb_multiclass_device_supported(self, device_kind) &
            result(supported)
        class(xgboost_multiclass_t), intent(in) :: self
        integer, intent(in) :: device_kind

        supported = self%initialized .and. device_kind == FORTML_DEVICE_CPU
    end function xgb_multiclass_device_supported

    logical function xgb_multiclass_fitted(self) result(fitted)
        class(xgboost_multiclass_t), intent(in) :: self

        fitted = self%initialized
    end function xgb_multiclass_fitted

    logical function valid_query(self, x) result(valid)
        class(xgboost_multiclass_t), intent(in) :: self
        real(dp), intent(in) :: x(:, :)

        valid = self%initialized
        if (.not. valid) return
        valid = size(x, 1) > 0 .and. size(x, 2) == self%n_inputs
        if (.not. valid) return
        valid = all(ieee_is_finite(x) .or. ieee_is_nan(x))
        if (.not. valid) return
        if (allocated(self%one_vs_rest)) then
            valid = self%one_vs_rest(1)%accepts_missing() .or. &
                .not. any(ieee_is_nan(x))
        else
            valid = .false.
        end if
    end function valid_query

    logical function valid_multiclass_model(self) result(valid)
        class(xgboost_multiclass_t), intent(in) :: self
        integer :: i, estimators

        valid = self%initialized
        if (.not. valid) return
        if (.not. allocated(self%class_label)) return
        if (.not. allocated(self%one_vs_rest)) return
        if (self%n_inputs < 1 .or. size(self%class_label) < 2) return
        if (size(self%one_vs_rest) /= size(self%class_label)) return
        do i = 2, size(self%class_label)
            if (self%class_label(i) <= self%class_label(i - 1)) return
        end do
        estimators = 0
        do i = 1, size(self%one_vs_rest)
            if (.not. self%one_vs_rest(i)%fitted()) return
            if (self%one_vs_rest(i)%feature_count() /= self%n_inputs) return
            if (trim(self%one_vs_rest(i)%objective_name()) /= "logistic") return
            if (i == 1) then
                estimators = self%one_vs_rest(i)%estimator_count()
                if (estimators < 1) return
            else if (self%one_vs_rest(i)%estimator_count() /= estimators) then
                return
            end if
        end do
        valid = .true.
    end function valid_multiclass_model

    subroutine multi_write_i(unit, key, value, ios)
        integer, intent(in) :: unit, value
        character(*), intent(in) :: key
        integer, intent(out) :: ios

        write(unit, "(A,1X,I0)", iostat=ios) trim(key), value
    end subroutine multi_write_i

    subroutine multi_write_r(unit, key, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: key
        real(dp), intent(in) :: value
        integer, intent(out) :: ios

        write(unit, "(A,1X,ES24.16E3)", iostat=ios) trim(key), value
    end subroutine multi_write_r

    subroutine multi_write_l(unit, key, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: key
        logical, intent(in) :: value
        integer, intent(out) :: ios

        write(unit, "(A,1X,I0)", iostat=ios) trim(key), merge(1, 0, value)
    end subroutine multi_write_l

    subroutine multi_read_i(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        integer, intent(out) :: value
        integer, intent(out) :: ios
        character(len=80) :: key

        read(unit, *, iostat=ios) key, value
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
    end subroutine multi_read_i

    subroutine multi_read_r(unit, expected, value, ios)
        integer, intent(in) :: unit
        character(*), intent(in) :: expected
        real(dp), intent(out) :: value
        integer, intent(out) :: ios
        character(len=80) :: key

        read(unit, *, iostat=ios) key, value
        if (ios == 0 .and. trim(key) /= trim(expected)) ios = 1
    end subroutine multi_read_r

    subroutine multi_read_l(unit, expected, value, ios)
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
    end subroutine multi_read_l

    subroutine make_child_path(path, index, child_path, ios)
        character(*), intent(in) :: path
        integer, intent(in) :: index
        character(*), intent(out) :: child_path
        integer, intent(out) :: ios

        child_path = ""
        write(child_path, '(A,".xgb-child-",I0)', iostat=ios) trim(path), index
    end subroutine make_child_path

    subroutine delete_file(path)
        character(*), intent(in) :: path
        integer :: unit, ios

        open(newunit=unit, file=path, status="old", action="read", iostat=ios)
        if (ios == 0) close(unit, status="delete", iostat=ios)
    end subroutine delete_file

    subroutine unique_sorted_labels(labels, classes)
        integer, intent(in) :: labels(:)
        integer, allocatable, intent(out) :: classes(:)
        integer :: i, j, n_unique, temporary

        allocate(classes(size(labels)))
        classes = labels
        do i = 2, size(classes)
            temporary = classes(i)
            j = i - 1
            do while (j >= 1)
                if (classes(j) <= temporary) exit
                classes(j + 1) = classes(j)
                j = j - 1
            end do
            classes(j + 1) = temporary
        end do
        n_unique = 1
        do i = 2, size(classes)
            if (classes(i) /= classes(n_unique)) then
                n_unique = n_unique + 1
                classes(n_unique) = classes(i)
            end if
        end do
        if (n_unique < size(classes)) then
            classes = classes(:n_unique)
        end if
    end subroutine unique_sorted_labels

    real(dp) function multiclass_log_loss(probabilities, totals, labels, classes, &
            observation_weight, weight_sum) result(loss)
        real(dp), intent(in) :: probabilities(:, :), totals(:)
        integer, intent(in) :: labels(:), classes(:)
        real(dp), intent(in) :: observation_weight(:), weight_sum
        integer :: i, class_index
        real(dp), parameter :: probability_floor = 1.0e-15_dp

        loss = 0.0_dp
        do i = 1, size(labels)
            class_index = find_label_index(classes, labels(i))
            if (class_index < 1 .or. class_index > size(probabilities, 2)) then
                loss = huge(1.0_dp)
                return
            end if
            loss = loss - observation_weight(i)*log(max( &
                probabilities(i, class_index)/totals(i), probability_floor))
        end do
        loss = loss/weight_sum
    end function multiclass_log_loss

    integer function find_label_index(classes, label) result(index)
        integer, intent(in) :: classes(:), label
        integer :: i

        index = 0
        do i = 1, size(classes)
            if (classes(i) == label) then
                index = i
                return
            end if
        end do
    end function find_label_index

end module fortml_xgboost_multiclass
