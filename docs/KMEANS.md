# Dense k-means

FortML's `kmeans_t` is a small, deterministic dense-clustering baseline. It
is intended to be useful in a reproducible CPU workflow while making the
boundaries of the current implementation explicit.

## Contract

Samples are rows of a finite, nonempty `(n_samples,n_features)` matrix. The
fit defaults are `n_clusters=8`, `max_iter=300`, `tolerance=1e-4`, and
`initialization_seed=1`. Initial centers are selected from a cyclic seeded
sample of rows, then Lloyd updates use a lowest-index tie break. The same
inputs and options therefore produce the same labels and centers across runs.
The initialization is deliberately documented as a deterministic baseline; it
is not k-means++ and should not be presented as full scikit-learn parity.

An empty cluster is a `FORTNUM_CONVERGENCE_ERROR`. FortML does not silently
reseed it, since doing so would change the persisted fit and its derivative
boundary. Reaching `max_iter` before the tolerance is met returns the same
status but retains the last finite state. Invalid dimensions, nonfinite values,
and malformed options return `FORTNUM_DOMAIN_ERROR`.

`predict` returns the nearest-center integer label. `transform` returns the
Euclidean distance from every sample to every center, and `inertia` is the sum
of squared distances for the final assignments. `fit_transform` combines the
two operations. The fitted center/label arrays and scalar metadata are exposed
through copy-returning accessors.

## Derivatives and device behavior

`transform_jvp` and `transform_vjp` differentiate distances with centers held
fixed. They are exact away from zero distances and return a typed domain error
at a zero distance, where the Euclidean norm is nonsmooth. Fit-time Lloyd
assignments and integer labels are discrete and have no derivative contract.
CUDA/device-resident fit, prediction, and transformation currently return
`FORTNUM_NOT_IMPLEMENTED`; the device argument never causes a hidden CPU
fallback.

The independent `test/test_kmeans.f90` checks centroid, label, inertia, and
distance values against a hand-computed six-point oracle, verifies the
fixed-center JVP/VJP adjoint relation, and exercises empty-cluster, nonfinite,
and CUDA refusal paths. The release workload is
`app/fortml_bench_kmeans.f90`; its output is consumed by the companion
`fortml-bench/scripts/bench_kmeans.py` lane, which independently recomputes
the fixture and inertia before recording timing.

