!> Capability and estimator-tag contracts shared by model and transform APIs.
module fortml_estimator_capabilities
    use fortnum_status, only: fortnum_status_t, status_set, FORTNUM_OK, &
        FORTNUM_DOMAIN_ERROR
    implicit none
    private

    integer, parameter, public :: FORTML_ROLE_NONE = 0
    integer, parameter, public :: FORTML_ROLE_TRANSFORMER = 1
    integer, parameter, public :: FORTML_ROLE_PREDICTOR = 2
    integer, parameter, public :: FORTML_ROLE_REGRESSOR = 4
    integer, parameter, public :: FORTML_ROLE_CLASSIFIER = 8

    integer, parameter, public :: FORTML_INPUT_DENSE = 1
    integer, parameter, public :: FORTML_INPUT_SPARSE = 2
    integer, parameter, public :: FORTML_INPUT_MISSING = 3
    integer, parameter, public :: FORTML_INPUT_SAMPLE_WEIGHT = 4
    integer, parameter, public :: FORTML_INPUT_PARTIAL_FIT = 5

    integer, parameter, public :: FORTML_DERIVATIVE_INPUT_JVP = 1
    integer, parameter, public :: FORTML_DERIVATIVE_INPUT_VJP = 2
    integer, parameter, public :: FORTML_DERIVATIVE_INPUT_HVP = 3
    integer, parameter, public :: FORTML_DERIVATIVE_PARAMETER_JVP = 4
    integer, parameter, public :: FORTML_DERIVATIVE_PARAMETER_VJP = 5
    integer, parameter, public :: FORTML_DERIVATIVE_PARAMETER_HVP = 6
    integer, parameter, public :: FORTML_DERIVATIVE_HYPERPARAMETER_JVP = 7
    integer, parameter, public :: FORTML_DERIVATIVE_HYPERPARAMETER_VJP = 8
    integer, parameter, public :: FORTML_DERIVATIVE_HYPERPARAMETER_HVP = 9

    integer, parameter, public :: FORTML_CAPABILITY_DEVICE_CPU = 0
    integer, parameter, public :: FORTML_CAPABILITY_DEVICE_CUDA = 1
    integer, parameter, public :: FORTML_CAPABILITY_DEVICE_OPENACC = 2

    integer, parameter :: CAPABILITY_NAME_LENGTH = 96

    !> A machine-readable contract for one fitted or configurable estimator.
    !>
    !> The record is intentionally a value object.  Model implementations can
    !> return it without exposing private state, while generic pipeline and
    !> validation code can query the same tags without type-selecting every
    !> estimator.  A false tag is a declared boundary: callers must not infer
    !> support from a similarly named model method.
    type, public :: estimator_capability_t
        character(len=CAPABILITY_NAME_LENGTH) :: name = ""
        integer :: roles = FORTML_ROLE_NONE
        integer :: n_features_in = 0
        integer :: n_features_out = 0
        integer :: n_targets = 0
        integer :: n_classes = 0
        logical :: fitted = .false.

        logical :: supports_dense = .false.
        logical :: supports_sparse = .false.
        logical :: supports_missing = .false.
        logical :: supports_sample_weight = .false.
        logical :: supports_partial_fit = .false.

        logical :: supports_transform = .false.
        logical :: supports_predict = .false.
        logical :: supports_decision_function = .false.
        logical :: supports_predict_proba = .false.

        logical :: supports_input_jvp = .false.
        logical :: supports_input_vjp = .false.
        logical :: supports_input_hvp = .false.
        logical :: supports_parameter_jvp = .false.
        logical :: supports_parameter_vjp = .false.
        logical :: supports_parameter_hvp = .false.
        logical :: supports_hyperparameter_jvp = .false.
        logical :: supports_hyperparameter_vjp = .false.
        logical :: supports_hyperparameter_hvp = .false.

        logical :: supports_cpu = .false.
        logical :: supports_cuda = .false.
        logical :: supports_openacc = .false.
        logical :: supports_resident = .false.
    contains
        procedure, public :: initialize => capability_initialize
        procedure, public :: valid => capability_valid
        procedure, public :: validate => capability_validate
        procedure, public :: is_fitted => capability_is_fitted
        procedure, public :: has_role => capability_has_role
        procedure, public :: supports_input => capability_supports_input
        procedure, public :: supports_derivative => capability_supports_derivative
        procedure, public :: supports_device => capability_supports_device
        procedure, public :: satisfies => capability_satisfies
        procedure, public :: require => capability_require
        procedure, public :: feature_count => capability_feature_count
        procedure, public :: output_count => capability_output_count
    end type estimator_capability_t

    public :: make_transformer_capabilities
    public :: make_predictor_capabilities
    public :: make_regressor_capabilities
    public :: make_classifier_capabilities
    public :: require_estimator_capability

contains

    !> Initialize a generic capability record.
    subroutine capability_initialize(self, name, roles, n_features_in, status, &
            n_features_out, n_targets, n_classes, fitted)
        class(estimator_capability_t), intent(out) :: self
        character(*), intent(in) :: name
        integer, intent(in) :: roles, n_features_in
        type(fortnum_status_t), intent(out) :: status
        integer, intent(in), optional :: n_features_out, n_targets, n_classes
        logical, intent(in), optional :: fitted

        self%name = ""
        self%roles = FORTML_ROLE_NONE
        self%n_features_in = 0
        self%n_features_out = 0
        self%n_targets = 0
        self%n_classes = 0
        self%fitted = .false.
        self%supports_dense = .false.
        self%supports_sparse = .false.
        self%supports_missing = .false.
        self%supports_sample_weight = .false.
        self%supports_partial_fit = .false.
        self%supports_transform = .false.
        self%supports_predict = .false.
        self%supports_decision_function = .false.
        self%supports_predict_proba = .false.
        self%supports_input_jvp = .false.
        self%supports_input_vjp = .false.
        self%supports_input_hvp = .false.
        self%supports_parameter_jvp = .false.
        self%supports_parameter_vjp = .false.
        self%supports_parameter_hvp = .false.
        self%supports_hyperparameter_jvp = .false.
        self%supports_hyperparameter_vjp = .false.
        self%supports_hyperparameter_hvp = .false.
        self%supports_cpu = .false.
        self%supports_cuda = .false.
        self%supports_openacc = .false.
        self%supports_resident = .false.
        if (len_trim(name) < 1 .or. len_trim(name) > CAPABILITY_NAME_LENGTH) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "estimator capability: name is empty or too long")
            return
        end if
        if (roles <= FORTML_ROLE_NONE .or. iand(roles, not( &
                FORTML_ROLE_TRANSFORMER + FORTML_ROLE_PREDICTOR + &
                FORTML_ROLE_REGRESSOR + FORTML_ROLE_CLASSIFIER)) /= 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "estimator capability: role mask is invalid")
            return
        end if
        if (n_features_in < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "estimator capability: n_features_in must be positive")
            return
        end if
        self%name = trim(name)
        self%roles = roles
        self%n_features_in = n_features_in
        if (present(n_features_out)) self%n_features_out = n_features_out
        if (present(n_targets)) self%n_targets = n_targets
        if (present(n_classes)) self%n_classes = n_classes
        if (present(fitted)) self%fitted = fitted
        if (self%n_features_out < 0 .or. self%n_targets < 0 .or. &
                self%n_classes < 0) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "estimator capability: output counts must be nonnegative")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine capability_initialize

    function make_transformer_capabilities(name, n_features_in, n_features_out, &
            status, fitted) result(capability)
        character(*), intent(in) :: name
        integer, intent(in) :: n_features_in, n_features_out
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: fitted
        type(estimator_capability_t) :: capability

        call capability%initialize(name, FORTML_ROLE_TRANSFORMER, n_features_in, &
            status, n_features_out=n_features_out, fitted=fitted)
        if (status%code /= FORTNUM_OK) return
        capability%supports_dense = .true.
        capability%supports_transform = .true.
        capability%supports_cpu = .true.
    end function make_transformer_capabilities

    function make_predictor_capabilities(name, n_features_in, n_targets, status, &
            fitted) result(capability)
        character(*), intent(in) :: name
        integer, intent(in) :: n_features_in, n_targets
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: fitted
        type(estimator_capability_t) :: capability

        call capability%initialize(name, FORTML_ROLE_PREDICTOR, n_features_in, &
            status, n_targets=n_targets, fitted=fitted)
        if (status%code /= FORTNUM_OK) return
        capability%supports_dense = .true.
        capability%supports_predict = .true.
        capability%supports_cpu = .true.
    end function make_predictor_capabilities

    function make_regressor_capabilities(name, n_features_in, n_targets, status, &
            fitted) result(capability)
        character(*), intent(in) :: name
        integer, intent(in) :: n_features_in, n_targets
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: fitted
        type(estimator_capability_t) :: capability

        call capability%initialize(name, ior(FORTML_ROLE_PREDICTOR, &
            FORTML_ROLE_REGRESSOR), n_features_in, status, n_targets=n_targets, &
            fitted=fitted)
        if (status%code /= FORTNUM_OK) return
        capability%supports_dense = .true.
        capability%supports_predict = .true.
        capability%supports_decision_function = .true.
        capability%supports_cpu = .true.
    end function make_regressor_capabilities

    function make_classifier_capabilities(name, n_features_in, n_classes, status, &
            fitted, n_targets) result(capability)
        character(*), intent(in) :: name
        integer, intent(in) :: n_features_in, n_classes
        type(fortnum_status_t), intent(out) :: status
        logical, intent(in), optional :: fitted
        integer, intent(in), optional :: n_targets
        type(estimator_capability_t) :: capability
        integer :: targets

        targets = 1
        if (present(n_targets)) targets = n_targets
        call capability%initialize(name, ior(FORTML_ROLE_PREDICTOR, &
            FORTML_ROLE_CLASSIFIER), n_features_in, status, n_targets=targets, &
            n_classes=n_classes, fitted=fitted)
        if (status%code /= FORTNUM_OK) return
        capability%supports_dense = .true.
        capability%supports_predict = .true.
        capability%supports_decision_function = .true.
        capability%supports_predict_proba = .true.
        capability%supports_cpu = .true.
    end function make_classifier_capabilities

    logical function capability_valid(self) result(valid)
        class(estimator_capability_t), intent(in) :: self

        valid = len_trim(self%name) > 0 .and. len_trim(self%name) <= &
            CAPABILITY_NAME_LENGTH .and. self%roles > FORTML_ROLE_NONE .and. &
            iand(self%roles, not(FORTML_ROLE_TRANSFORMER + &
                FORTML_ROLE_PREDICTOR + FORTML_ROLE_REGRESSOR + &
                FORTML_ROLE_CLASSIFIER)) == 0 .and. &
            self%n_features_in > 0 .and. self%n_features_out >= 0 .and. &
            self%n_targets >= 0 .and. self%n_classes >= 0
    end function capability_valid

    subroutine capability_validate(self, status)
        class(estimator_capability_t), intent(in) :: self
        type(fortnum_status_t), intent(out) :: status

        if (.not. self%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "estimator capability: record is invalid")
            return
        end if
        if (self%has_role(FORTML_ROLE_CLASSIFIER) .and. &
                .not. self%has_role(FORTML_ROLE_PREDICTOR)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "estimator capability: classifier needs predictor role")
            return
        end if
        if (self%has_role(FORTML_ROLE_REGRESSOR) .and. &
                .not. self%has_role(FORTML_ROLE_PREDICTOR)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "estimator capability: regressor needs predictor role")
            return
        end if
        if (self%has_role(FORTML_ROLE_CLASSIFIER) .and. &
                self%n_classes < 2) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "estimator capability: classifier needs at least two classes")
            return
        end if
        if (self%has_role(FORTML_ROLE_REGRESSOR) .and. self%n_targets < 1) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "estimator capability: regressor needs a target count")
            return
        end if
        if (self%supports_transform .and. &
                .not. self%has_role(FORTML_ROLE_TRANSFORMER)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "estimator capability: transform tag requires transformer role")
            return
        end if
        if ((self%supports_predict .or. self%supports_predict_proba .or. &
                self%supports_decision_function) .and. &
                .not. self%has_role(FORTML_ROLE_PREDICTOR)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "estimator capability: prediction tag requires predictor role")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine capability_validate

    logical function capability_is_fitted(self) result(value)
        class(estimator_capability_t), intent(in) :: self

        value = self%fitted
    end function capability_is_fitted

    logical function capability_has_role(self, role) result(value)
        class(estimator_capability_t), intent(in) :: self
        integer, intent(in) :: role

        value = role > FORTML_ROLE_NONE .and. iand(self%roles, role) == role
    end function capability_has_role

    logical function capability_supports_input(self, input_kind) result(value)
        class(estimator_capability_t), intent(in) :: self
        integer, intent(in) :: input_kind

        select case (input_kind)
        case (FORTML_INPUT_DENSE)
            value = self%supports_dense
        case (FORTML_INPUT_SPARSE)
            value = self%supports_sparse
        case (FORTML_INPUT_MISSING)
            value = self%supports_missing
        case (FORTML_INPUT_SAMPLE_WEIGHT)
            value = self%supports_sample_weight
        case (FORTML_INPUT_PARTIAL_FIT)
            value = self%supports_partial_fit
        case default
            value = .false.
        end select
    end function capability_supports_input

    logical function capability_supports_derivative(self, derivative_kind) &
            result(value)
        class(estimator_capability_t), intent(in) :: self
        integer, intent(in) :: derivative_kind

        select case (derivative_kind)
        case (FORTML_DERIVATIVE_INPUT_JVP)
            value = self%supports_input_jvp
        case (FORTML_DERIVATIVE_INPUT_VJP)
            value = self%supports_input_vjp
        case (FORTML_DERIVATIVE_INPUT_HVP)
            value = self%supports_input_hvp
        case (FORTML_DERIVATIVE_PARAMETER_JVP)
            value = self%supports_parameter_jvp
        case (FORTML_DERIVATIVE_PARAMETER_VJP)
            value = self%supports_parameter_vjp
        case (FORTML_DERIVATIVE_PARAMETER_HVP)
            value = self%supports_parameter_hvp
        case (FORTML_DERIVATIVE_HYPERPARAMETER_JVP)
            value = self%supports_hyperparameter_jvp
        case (FORTML_DERIVATIVE_HYPERPARAMETER_VJP)
            value = self%supports_hyperparameter_vjp
        case (FORTML_DERIVATIVE_HYPERPARAMETER_HVP)
            value = self%supports_hyperparameter_hvp
        case default
            value = .false.
        end select
    end function capability_supports_derivative

    logical function capability_supports_device(self, device_kind) result(value)
        class(estimator_capability_t), intent(in) :: self
        integer, intent(in) :: device_kind

        select case (device_kind)
        case (FORTML_CAPABILITY_DEVICE_CPU)
            value = self%supports_cpu
        case (FORTML_CAPABILITY_DEVICE_CUDA)
            value = self%supports_cuda
        case (FORTML_CAPABILITY_DEVICE_OPENACC)
            value = self%supports_openacc
        case default
            value = .false.
        end select
    end function capability_supports_device

    logical function capability_satisfies(self, requirement) result(value)
        class(estimator_capability_t), intent(in) :: self
        type(estimator_capability_t), intent(in) :: requirement

        value = self%valid() .and. requirement%valid()
        if (.not. value) return
        value = requirement%roles == FORTML_ROLE_NONE .or. &
            self%has_role(requirement%roles)
        if (.not. value) return
        if (requirement%n_features_in > 0) then
            value = self%n_features_in == requirement%n_features_in
        end if
        if (.not. value) return
        if (requirement%n_features_out > 0) then
            value = self%n_features_out == requirement%n_features_out
        end if
        if (.not. value) return
        if (requirement%n_targets > 0) value = self%n_targets >= requirement%n_targets
        if (.not. value) return
        if (requirement%n_classes > 0) value = self%n_classes >= requirement%n_classes
        if (.not. value) return
        if (requirement%fitted) value = self%fitted
        if (.not. value) return

        value = capability_bool_requirements_satisfied(self, requirement)
    end function capability_satisfies

    logical function capability_bool_requirements_satisfied(self, requirement) &
            result(value)
        class(estimator_capability_t), intent(in) :: self
        type(estimator_capability_t), intent(in) :: requirement
        integer :: i

        value = .true.
        do i = 1, FORTML_INPUT_PARTIAL_FIT
            if (requirement%supports_input(i) .and. &
                    .not. self%supports_input(i)) then
                value = .false.
                return
            end if
        end do
        do i = 1, FORTML_DERIVATIVE_HYPERPARAMETER_HVP
            if (requirement%supports_derivative(i) .and. &
                    .not. self%supports_derivative(i)) then
                value = .false.
                return
            end if
        end do
        if (requirement%supports_transform .and. .not. self%supports_transform) then
            value = .false.
            return
        end if
        if (requirement%supports_predict .and. .not. self%supports_predict) then
            value = .false.
            return
        end if
        if (requirement%supports_decision_function .and. &
                .not. self%supports_decision_function) then
            value = .false.
            return
        end if
        if (requirement%supports_predict_proba .and. &
                .not. self%supports_predict_proba) then
            value = .false.
            return
        end if
        if (requirement%supports_cpu .and. .not. self%supports_cpu) then
            value = .false.
            return
        end if
        if (requirement%supports_cuda .and. .not. self%supports_cuda) then
            value = .false.
            return
        end if
        if (requirement%supports_openacc .and. .not. self%supports_openacc) then
            value = .false.
            return
        end if
        if (requirement%supports_resident .and. .not. self%supports_resident) then
            value = .false.
            return
        end if
    end function capability_bool_requirements_satisfied

    subroutine capability_require(self, requirement, status)
        class(estimator_capability_t), intent(in) :: self
        type(estimator_capability_t), intent(in) :: requirement
        type(fortnum_status_t), intent(out) :: status

        if (.not. requirement%valid()) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "estimator capability: requirement is invalid")
            return
        end if
        if (.not. self%satisfies(requirement)) then
            call status_set(status, FORTNUM_DOMAIN_ERROR, &
                "estimator capability: requirement is not supported")
            return
        end if
        call status_set(status, FORTNUM_OK, "")
    end subroutine capability_require

    subroutine require_estimator_capability(actual, requirement, status)
        type(estimator_capability_t), intent(in) :: actual, requirement
        type(fortnum_status_t), intent(out) :: status

        call actual%require(requirement, status)
    end subroutine require_estimator_capability

    integer function capability_feature_count(self) result(count)
        class(estimator_capability_t), intent(in) :: self

        count = self%n_features_in
    end function capability_feature_count

    integer function capability_output_count(self) result(count)
        class(estimator_capability_t), intent(in) :: self

        count = self%n_features_out
        if (self%has_role(FORTML_ROLE_CLASSIFIER)) count = self%n_classes
        if (count == 0 .and. self%has_role(FORTML_ROLE_REGRESSOR)) count = self%n_targets
        if (count == 0 .and. self%has_role(FORTML_ROLE_PREDICTOR)) count = self%n_targets
    end function capability_output_count

end module fortml_estimator_capabilities
