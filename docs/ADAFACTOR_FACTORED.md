# Layout-aware factored Adafactor

`fortml_adafactor_factored` implements the matrix-state Adafactor recurrence
for an explicitly described packed parameter layout. Matrix blocks use the
row/column estimator

```text
row <- decay row + (1-decay) mean_columns(gradient**2)
column <- decay column + (1-decay) mean_rows(gradient**2)
v[i,j] <- row[i] column[j] / mean(row)
```

One-dimensional blocks use the exact unfactored second-moment recurrence.
The update-RMS clip, optional relative step, and optional parameter scaling
are shared across all blocks, so the packed update has the same deterministic
semantics as `fortml_adafactor` while storing `O(rows+columns)` state for a
matrix block instead of `O(rows*columns)`.

## API

Construct one `adafactor_block_spec_t` for every contiguous region of the
packed vector. `first:last` must cover the vector in order and
`rows*columns` must equal the block length. Set `factored=.true.` only for
matrix blocks. `mlp_train` derives this layout from `mlp_t%parameter_layout()`
when `options%adafactor_factored=.true.`; dense weights are factored and bias
or singleton-dimension blocks remain vector states.

```fortran
use fortml_adafactor_factored, only: adafactor_factored_t

call optimizer%initialize(n_parameters, blocks, status, &
    learning_rate=1.0e-3_dp, decay=0.999_dp, epsilon=1.0e-30_dp)
call optimizer%step(parameters, gradient, status)
call optimizer%dense_second_moment(second_moment, status)
```

The independent `test_mlp_adafactor_factored` fixture compares three updates
against a separate row/column and vector recurrence, checks the dense-state
reconstruction, exercises the MLP trainer integration, and compares
uninterrupted training with native and formatted checkpoint resumes.

## Explicit boundaries

The recurrence is CPU-resident. `step_device` reports
`FORTNUM_NOT_IMPLEMENTED` for CUDA; it never copies a host vector and calls
that GPU execution. Factored checkpoints include block shape metadata and
flattened row, column, and vector states, and the resume path rejects a
different parameter layout before restoring any state. Formatted checkpoints
use schema 11; the default vector Adafactor path remains checkpoint/resume
compatible as before. These are typed boundaries, not silent fallbacks.
