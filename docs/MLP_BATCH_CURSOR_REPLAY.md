# Deterministic MLP data-loader cursors

`mlp_batch_iterator_t` exposes explicit mini-batch boundaries and now has a
small cursor value for deterministic replay.  `capture` records the sample
count, batch shape, epoch and position, the current shuffled permutation, and
the shuffle generator state.  `restore` validates that complete snapshot
before replacing an iterator, so a malformed cursor cannot partially mutate a
running loader.

```fortran
type(mlp_batch_iterator_t) :: loader, branch
type(mlp_batch_iterator_cursor_t) :: cursor
type(fortnum_status_t) :: status
integer, allocatable :: indices(:)
logical :: has_batch

call loader%initialize(1024, status, batch_size=32, shuffle=.true., seed=42)
call loader%reset(status)
call loader%next_batch(indices, has_batch, status)
call loader%capture(cursor, status)

! A second worker or validation replay can resume the exact suffix.
call branch%restore(cursor, status)
call branch%next_batch(indices, has_batch, status)
```

The cursor includes the permutation rather than only a seed.  Replaying after
an interruption therefore does not depend on a particular random-shuffle
implementation or on how many batches a worker has already consumed.  The
same cursor can be restored into a fresh iterator with different constructor
arguments; the captured batch shape and stream are authoritative.

The API is a host-side reference contract.  `mlp_batch_iterator_device_supported`
returns true for CPU and false for CUDA, while
`mlp_batch_iterator_require_device(FORTML_DEVICE_CUDA, status)` returns
`FORTNUM_NOT_IMPLEMENTED`.  There is no hidden host fallback for a requested
resident CUDA loader; a future resident implementation must provide an
explicit permutation/RNG and transfer contract first.

`test_mlp_batch_iterator` supplies an independent known-seed permutation
oracle, verifies suffix identity after capture/restore, checks transactional
refusal of a cleared cursor, and exercises the CPU/CUDA capability boundary.

