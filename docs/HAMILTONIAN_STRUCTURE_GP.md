# Separable Hamiltonian finite-feature GP initialization

`fortml_hamiltonian_structure_gp` provides a bounded warm start for the
separable Hamiltonian model

```text
H(q,p) = V(q) + T(p),     f(q,p) = (dT/dp, -dV/dq).
```

`hamiltonian_structure_gp_initializer_t%fit` fits two independent finite-
feature last-layer kernel-ridge posteriors: `(q, potential_target)` for `V`
and `(p, kinetic_target)` for `T`. `fit_apply` performs the fit and validated
application transactionally. Only the final affine layer of each scalar MLP
is changed. The hidden parameter snapshots and packed parameter counts are
checked before every apply or prediction; a changed hidden state or topology
returns `FORTNUM_DOMAIN_ERROR` without applying either posterior.

```fortran
use fortml_hamiltonian_structure_gp, only: &
    hamiltonian_structure_gp_initializer_t
type(hamiltonian_structure_gp_initializer_t) :: initializer
call initializer%fit_apply(model, q, v_target, p, t_target, status, &
    regularization=1.0e-3_dp)
call initializer%predict_components(model, q, p, v_mean, t_mean, status)
```

`metadata()` records the two sample/feature dimensions, regularization,
potential and kinetic training RMSE, hidden parameter counts, and
`structure_defect`. The latter is zero for a successful application because
separability is preserved by construction; it is a topology/state certificate,
not a claim that the finite network equals an infinite-width GP. The metadata
sets `exact_infinite_width=.false.` and `cuda_supported=.false.` explicitly.
`apply_cuda` and `predict_cuda` return typed `FORTNUM_NOT_IMPLEMENTED` without
host fallback or output mutation.

The independent normal-equation oracle and structure/refusal checks are in
`test/test_hamiltonian_structure_gp.f90`. The release benchmark application
is `fortml_bench_hamiltonian_structure_gp`; its CSV row reports fit/prediction
timings, both posterior RMSEs, and the structure defect. This slice does not
implement NNGP covariance propagation, sampled posterior weights, general
nonseparable Hamiltonians, symplectic GP priors, or resident GPU training.
