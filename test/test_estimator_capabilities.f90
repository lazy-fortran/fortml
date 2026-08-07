program test_estimator_capabilities
    use fortnum_kinds, only: dp
    use fortnum_status, only: fortnum_status_t, status_ok
    use fortml_estimator_capabilities, only: estimator_capability_t, &
        make_transformer_capabilities, make_regressor_capabilities, &
        make_classifier_capabilities, require_estimator_capability, &
        FORTML_ROLE_TRANSFORMER, FORTML_ROLE_PREDICTOR, FORTML_ROLE_REGRESSOR, &
        FORTML_ROLE_CLASSIFIER, &
        FORTML_INPUT_DENSE, FORTML_INPUT_SPARSE, &
        FORTML_DERIVATIVE_INPUT_JVP, FORTML_DERIVATIVE_PARAMETER_HVP, &
        FORTML_CAPABILITY_DEVICE_CPU, FORTML_CAPABILITY_DEVICE_CUDA
    use fortml_validation, only: validate_estimator_capability
    use fortml_basis, only: basis_map_t, make_polynomial_basis
    use fortml_pipeline, only: basis_pipeline_t, make_basis_pipeline
    implicit none

    integer :: failures

    failures = 0
    call test_roles_and_queries(failures)
    call test_requirements_and_refusals(failures)
    call test_pipeline_contract(failures)
    if (failures > 0) error stop "estimator capability tests failed"
    write (*, '(a)') "PASS estimator capability independent behavioral oracles"

contains

    subroutine check(condition, label, failures)
        logical, intent(in) :: condition
        character(*), intent(in) :: label
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (*, '(a)') "  FAIL "//label
        end if
    end subroutine check

    subroutine test_roles_and_queries(failures)
        integer, intent(inout) :: failures
        type(estimator_capability_t) :: regressor, classifier
        type(fortnum_status_t) :: status

        regressor = make_regressor_capabilities("ridge", 3, 1, status, fitted=.true.)
        call check(status_ok(status), "regressor constructor", failures)
        call check(regressor%valid(), "regressor validity", failures)
        call check(regressor%is_fitted(), "fitted tag", failures)
        call check(regressor%has_role(FORTML_ROLE_REGRESSOR), &
            "regressor role", failures)
        call check(regressor%supports_input(FORTML_INPUT_DENSE), &
            "dense input tag", failures)
        call check(.not. regressor%supports_input(FORTML_INPUT_SPARSE), &
            "sparse refusal tag", failures)
        call check(regressor%supports_device(FORTML_CAPABILITY_DEVICE_CPU), &
            "CPU device tag", failures)
        call check(.not. regressor%supports_device(FORTML_CAPABILITY_DEVICE_CUDA), &
            "CUDA refusal tag", failures)

        classifier = make_classifier_capabilities("softmax", 3, 4, status, &
            fitted=.false.)
        call check(status_ok(status), "classifier constructor", failures)
        call check(classifier%has_role(FORTML_ROLE_CLASSIFIER) .and. &
            classifier%has_role(FORTML_ROLE_PREDICTOR), &
            "classifier predictor roles", failures)
        call check(classifier%supports_predict_proba, &
            "classifier probability tag", failures)
        call check(classifier%output_count() == 4, &
            "classifier output count", failures)
        call validate_estimator_capability(classifier, status)
        call check(status_ok(status), "classifier validation", failures)
    end subroutine test_roles_and_queries

    subroutine test_requirements_and_refusals(failures)
        integer, intent(inout) :: failures
        type(estimator_capability_t) :: actual, requirement, bad
        type(fortnum_status_t) :: status

        actual = make_regressor_capabilities("ridge", 3, 1, status, fitted=.true.)
        actual%supports_parameter_hvp = .true.
        call requirement%initialize("regressor requirement", FORTML_ROLE_REGRESSOR, &
            3, status, n_targets=1, fitted=.true.)
        requirement%supports_dense = .true.
        requirement%supports_parameter_hvp = .true.
        call require_estimator_capability(actual, requirement, status)
        call check(status_ok(status), "satisfied capability requirement", failures)

        call requirement%initialize("CUDA requirement", FORTML_ROLE_REGRESSOR, 3, &
            status, n_targets=1)
        requirement%supports_cuda = .true.
        call require_estimator_capability(actual, requirement, status)
        call check(.not. status_ok(status), "unsupported device refusal", failures)

        bad = make_classifier_capabilities("bad", 3, 1, status)
        call validate_estimator_capability(bad, status)
        call check(.not. status_ok(status), "invalid classifier refusal", failures)
        call bad%initialize("bad-role", 16, 3, status)
        call check(.not. status_ok(status), "invalid role-mask refusal", failures)
        call check(actual%supports_derivative(FORTML_DERIVATIVE_PARAMETER_HVP), &
            "parameter HVP query", failures)
        call check(.not. actual%supports_derivative(FORTML_DERIVATIVE_INPUT_JVP), &
            "input JVP refusal tag", failures)
    end subroutine test_requirements_and_refusals

    subroutine test_pipeline_contract(failures)
        integer, intent(inout) :: failures
        type(basis_map_t) :: basis
        type(basis_pipeline_t) :: pipeline
        type(estimator_capability_t) :: capability
        type(fortnum_status_t) :: status
        real(dp) :: x(3, 2)
        real(dp), allocatable :: features(:, :)

        x = reshape([0.0_dp, 1.0_dp, 2.0_dp, 1.0_dp, 0.0_dp, -1.0_dp], &
            shape(x))
        basis = make_polynomial_basis(2, 2, status, include_intercept=.true.)
        call check(status_ok(status), "pipeline basis constructor", failures)
        pipeline = make_basis_pipeline(2, status)
        call check(status_ok(status), "pipeline constructor", failures)
        call pipeline%append(basis, status, name="quadratic")
        call check(status_ok(status), "pipeline append", failures)
        call pipeline%fit(x, status)
        call check(status_ok(status) .and. pipeline%is_fitted(), &
            "pipeline fit", failures)
        allocate(features(3, pipeline%feature_count()))
        call pipeline%transform(x, features, status)
        call check(status_ok(status), "pipeline transform", failures)
        call pipeline%capabilities(capability, status)
        call check(status_ok(status), "pipeline capability query", failures)
        call check(capability%has_role(FORTML_ROLE_TRANSFORMER), &
            "pipeline transformer role", failures)
        call check(capability%is_fitted() .and. capability%feature_count() == 2, &
            "pipeline fitted feature count", failures)
        call check(capability%output_count() == pipeline%feature_count(), &
            "pipeline output count", failures)
        call check(capability%supports_derivative(FORTML_DERIVATIVE_INPUT_JVP) .and. &
            capability%supports_derivative(FORTML_DERIVATIVE_PARAMETER_HVP), &
            "pipeline derivative tags", failures)
        call check(.not. capability%supports_predict, &
            "pipeline predictor refusal", failures)
    end subroutine test_pipeline_contract

end program test_estimator_capabilities
