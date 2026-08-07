program test_device_contract_new_features
    !! Device contract tests for the four recent product families.
    !!
    !! These checks deliberately use a synthetic selected CUDA context.  This
    !! exercises the refusal boundary without requiring a CUDA driver and
    !! ensures that an available unrelated CUDA kernel cannot cause a hidden
    !! host fallback for a model that has no resident implementation.
    use, intrinsic :: iso_fortran_env, only: real64, error_unit
    use fortnum_status, only: fortnum_status_t, status_ok, FORTNUM_NOT_IMPLEMENTED
    use fortml_device, only: fortml_device_t, FORTML_DEVICE_CPU, FORTML_DEVICE_CUDA
    use fortml_elastic_net_regression, only: elastic_net_regression_t
    use fortml_ovo_logistic_classifier, only: ovo_logistic_classifier_t
    use fortml_mlp_schedules, only: mlp_learning_rate_schedule_t, &
        make_mlp_schedule_cosine_decay
    use fortml_gp_classification, only: gp_classification_t, &
        gp_classification_likelihood_device_supported
    implicit none

    type(fortml_device_t) :: cpu, cuda
    type(fortnum_status_t) :: status
    type(elastic_net_regression_t) :: elastic_net
    type(ovo_logistic_classifier_t) :: ovo
    type(gp_classification_t) :: gp
    real(real64) :: x(2, 1), y(2, 1), mean(2), variance(2), probabilities(2, 2)
    real(real64) :: schedule_rate
    integer :: labels(2), failures

    failures = 0
    x = reshape([0.0_real64, 1.0_real64], shape(x))
    y = reshape([1.0_real64, 2.0_real64], shape(y))
    labels = [1, 2]

    call cpu%select(FORTML_DEVICE_CPU, status)
    call check(status_ok(status), "CPU device selection", failures)
    cuda%kind = FORTML_DEVICE_CUDA
    cuda%selected = .true.
    cuda%available = .true.

    call elastic_net%predict_device(cuda, x, y, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "elastic-net CUDA prediction refusal", failures)
    call check(.not. elastic_net%device_supported(FORTML_DEVICE_CUDA), &
        "elastic-net CUDA capability refusal", failures)
    call check(.not. elastic_net%device_supported(FORTML_DEVICE_CPU), &
        "unfitted elastic-net CPU capability refusal", failures)

    call ovo%predict_proba_device(cuda, x, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "OVO CUDA probability refusal", failures)
    call ovo%predict_device(cuda, x, labels, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "OVO CUDA label refusal", failures)
    call check(.not. ovo%device_supported(FORTML_DEVICE_CUDA), &
        "OVO CUDA capability refusal", failures)

    block
        type(mlp_learning_rate_schedule_t) :: schedule
        schedule = make_mlp_schedule_cosine_decay(10, 0.1_real64)
        call check(schedule%device_supported(FORTML_DEVICE_CPU), &
            "schedule CPU capability", failures)
        call check(.not. schedule%device_supported(FORTML_DEVICE_CUDA), &
            "schedule CUDA capability refusal", failures)
        call schedule%rate(2, 0.2_real64, schedule_rate, status)
        call check(status_ok(status), "schedule host reference remains usable", failures)
    end block

    call gp%predict_latent_device(cuda, x, mean, variance, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "GP CUDA latent prediction refusal", failures)
    call gp%predict_proba_device(cuda, x, probabilities, status)
    call check(status%code == FORTNUM_NOT_IMPLEMENTED, &
        "GP CUDA probability refusal", failures)
    call check(.not. gp%device_supported(FORTML_DEVICE_CUDA), &
        "GP CUDA capability refusal", failures)
    call check(gp_classification_likelihood_device_supported(FORTML_DEVICE_CPU), &
        "GP likelihood CPU capability", failures)
    call check(.not. gp_classification_likelihood_device_supported(FORTML_DEVICE_CUDA), &
        "GP likelihood CUDA derivative refusal", failures)

    if (failures > 0) then
        write (error_unit, '(a,i0)') "FAIL new-feature device contracts: ", failures
        error stop 1
    end if
    write (*, '(a)') "PASS new-feature device refusal contracts"

contains

    subroutine check(condition, description, failures)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: description
        integer, intent(inout) :: failures

        if (.not. condition) then
            failures = failures + 1
            write (error_unit, '(a)') "FAIL "//trim(description)
        end if
    end subroutine check

end program test_device_contract_new_features
