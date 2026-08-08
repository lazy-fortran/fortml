program fortml_bench_xgboost_multiclass_persistence
    !! Release workload for the portable OVR XGBoost multiclass snapshot.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_xgboost_multiclass, only: xgboost_multiclass_t
    use fortml_xgboost, only: xgboost_options_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 9, n_features = 1, n_classes = 3
    integer, parameter :: n_estimators = 3
    real(dp) :: x(n_samples, n_features), query(3, n_features)
    real(dp) :: before(3, n_classes), after(3, n_classes)
    integer :: labels(n_samples), classes(n_classes)
    integer(int64) :: clock_start, clock_end, clock_rate
    real(dp) :: elapsed, error, probability_sum
    type(xgboost_multiclass_t) :: model, restored
    type(xgboost_options_t) :: options
    type(fortnum_status_t) :: status
    character(*), parameter :: path = "fortml_bench_xgboost_multiclass.snapshot"

    x(:, 1) = [-4.0_dp, -3.0_dp, -2.0_dp, -1.0_dp, 0.0_dp, 1.0_dp, &
        2.0_dp, 3.0_dp, 4.0_dp]
    labels = [-8, -8, -8, 2, 2, 2, 11, 11, 11]
    query(:, 1) = [-2.3_dp, 0.1_dp, 2.4_dp]
    options%n_estimators = n_estimators
    options%max_depth = 1
    options%learning_rate = 0.4_dp
    options%l2 = 1.0_dp
    options%min_child_weight = 0.0_dp

    call model%fit(x, labels, status, options)
    if (.not. status_ok(status)) error stop "XGBoost multiclass persistence fit failed"
    call model%predict_proba(query, before, status)
    if (.not. status_ok(status)) error stop "XGBoost multiclass persistence prediction failed"
    classes = model%classes()

    call system_clock(clock_start, clock_rate)
    call model%save_text(path, status)
    call restored%load_text(path, status)
    call system_clock(clock_end)
    if (.not. status_ok(status)) error stop "XGBoost multiclass persistence round trip failed"
    call restored%predict_proba(query, after, status)
    if (.not. status_ok(status)) error stop "XGBoost multiclass restored prediction failed"
    elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp)
    error = maxval(abs(before - after))
    probability_sum = sum(after)

    write (*, '(a,i0,a,i0,a,i0,a,i0,a,es24.16,a,es24.16,a,es24.16,a,i0)') &
        "xgb_multiclass_persistence,round_trip,", n_samples, ",", n_features, &
        ",", n_classes, ",", n_estimators, ",", elapsed, ",", error, ",", &
        probability_sum, ",", sum(classes)
    call delete_file(path)

contains

    subroutine delete_file(name)
        character(*), intent(in) :: name
        integer :: unit, ios

        open(newunit=unit, file=name, status="old", action="read", iostat=ios)
        if (ios == 0) close(unit, status="delete", iostat=ios)
    end subroutine delete_file

end program fortml_bench_xgboost_multiclass_persistence
