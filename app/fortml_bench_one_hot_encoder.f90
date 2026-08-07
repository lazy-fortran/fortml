program fortml_bench_one_hot_encoder
    !! Release workload for the integer categorical one-hot encoder.
    !!
    !! The Python harness owns the independent dense NumPy oracle.  This app
    !! emits transformed values and fitted packed metadata when
    !! FORTML_BENCH_ONE_HOT_ORACLE is set, then reports release timings.
    use, intrinsic :: iso_fortran_env, only: dp => real64, int64
    use fortml_one_hot_encoder, only: one_hot_encoder_t
    use fortnum_status, only: fortnum_status_t, status_ok
    implicit none

    integer, parameter :: n_samples = 512, n_features = 5, n_query = 128
    integer, parameter :: fit_repetitions = 16, transform_repetitions = 128
    integer, parameter :: missing_value = -99
    integer :: train(n_samples, n_features), query(n_query, n_features)
    real(dp), allocatable :: transformed(:, :)
    integer, allocatable :: categories(:), category_offsets(:), output_offsets(:)
    integer :: output_count, j, k
    real(dp) :: fit_elapsed, transform_elapsed
    integer(int64) :: clock_start, clock_end, clock_rate
    type(one_hot_encoder_t) :: model
    type(fortnum_status_t) :: status
    character(len=1024) :: oracle_path
    integer :: oracle_unit, environment_status

    call make_fixture(train, query)
    call model%fit(train, status, handle_unknown="ignore", &
        missing_value=missing_value, handle_missing="category", drop_first=.true.)
    if (.not. status_ok(status)) error stop "one-hot benchmark fit failed"
    output_count = model%output_count()
    allocate(transformed(n_query, output_count))
    call model%transform(query, transformed, status)
    if (.not. status_ok(status)) error stop "one-hot benchmark transform failed"
    categories = model%categories()
    category_offsets = model%category_offsets()
    output_offsets = model%output_offsets()

    oracle_unit = -1
    call get_environment_variable("FORTML_BENCH_ONE_HOT_ORACLE", oracle_path, &
        status=environment_status)
    if (environment_status == 0 .and. len_trim(oracle_path) > 0) then
        open (newunit=oracle_unit, file=trim(oracle_path), status="replace", &
            action="write")
        write (oracle_unit, '(a)') "quantity,row,column,value"
        call write_oracle(oracle_unit, transformed, categories, category_offsets, &
            output_offsets)
        close (oracle_unit)
    end if
    if (oracle_only_requested()) stop

    call system_clock(clock_start, clock_rate)
    do k = 1, fit_repetitions
        call model%fit(train, status, handle_unknown="ignore", &
            missing_value=missing_value, handle_missing="category", drop_first=.true.)
        if (.not. status_ok(status)) error stop "one-hot timed fit failed"
    end do
    call system_clock(clock_end)
    fit_elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(fit_repetitions, dp)

    call system_clock(clock_start, clock_rate)
    do k = 1, transform_repetitions
        call model%transform(query, transformed, status)
        if (.not. status_ok(status)) error stop "one-hot timed transform failed"
    end do
    call system_clock(clock_end)
    transform_elapsed = real(clock_end - clock_start, dp)/real(clock_rate, dp) &
        /real(transform_repetitions, dp)

    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') &
        "one_hot_fit,", n_samples, ",", n_features, ",", output_count, ",", &
        fit_elapsed
    write (*, '(a,i0,a,i0,a,i0,a,es24.16)') &
        "one_hot_transform,", n_query, ",", n_features, ",", output_count, ",", &
        transform_elapsed

contains

    subroutine make_fixture(training, queries)
        integer, intent(out) :: training(:, :), queries(:, :)
        integer :: row, column

        do column = 1, n_features
            do row = 1, n_samples
                training(row, column) = mod(7*row + 3*column + row*column, &
                    column + 2)
            end do
        end do
        do row = 1, n_samples
            if (mod(row, 29) == 0) training(row, :) = missing_value
        end do
        do column = 1, n_features
            do row = 1, n_query
                queries(row, column) = mod(11*row + 5*column + row*column, &
                    column + 3)
            end do
        end do
        do row = 1, n_query
            if (mod(row, 17) == 0) queries(row, :) = missing_value
        end do
        do row = 1, n_query, 19
            queries(row, 3) = 1001
        end do
    end subroutine make_fixture

    subroutine write_oracle(unit, values, fitted_categories, fitted_category_offsets, &
            fitted_output_offsets)
        integer, intent(in) :: unit
        real(dp), intent(in) :: values(:, :)
        integer, intent(in) :: fitted_categories(:), fitted_category_offsets(:), &
            fitted_output_offsets(:)
        integer :: i, j

        do i = 1, size(values, 1)
            do j = 1, size(values, 2)
                write (unit, '(a,i0,a,i0,a,es26.17e3)') &
                    "transformed,", i, ",", j, ",", values(i, j)
            end do
        end do
        do i = 1, size(fitted_categories)
            write (unit, '(a,i0,a,i0)') "category,", i, ",1,", fitted_categories(i)
        end do
        do i = 1, size(fitted_category_offsets)
            write (unit, '(a,i0,a,i0)') "category_offset,", i, ",1,", &
                fitted_category_offsets(i)
        end do
        do i = 1, size(fitted_output_offsets)
            write (unit, '(a,i0,a,i0)') "output_offset,", i, ",1,", &
                fitted_output_offsets(i)
        end do
    end subroutine write_oracle

    logical function oracle_only_requested()
        character(len=16) :: value
        integer :: environment_status

        call get_environment_variable("FORTML_BENCH_ORACLE_ONLY", value, &
            status=environment_status)
        oracle_only_requested = environment_status == 0 .and. trim(value) == "1"
    end function oracle_only_requested

end program fortml_bench_one_hot_encoder
