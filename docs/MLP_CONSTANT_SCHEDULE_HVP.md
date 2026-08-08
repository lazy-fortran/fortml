# Affine constant-schedule outer HVP

`fortml_mlp_schedule_hypergradient` exposes an exact second-order product for
the useful linear baseline: a single dense MLP layer with a linear output
activation (there is no hidden activation to apply), trained by a fixed
constant learning-rate schedule.  The packed outer vector
is

```text
[ log(base_rate), log(l2), logit(min_rate_fraction), logit(decay_factor) ]
```

The last two coordinates are inactive for a constant schedule and therefore
have exact zero gradient and HVP components.  The first two coordinates are
positive transforms, so the recurrence remains valid for every finite trial
point accepted by the objective.

The implementation propagates first and mixed second parameter tangents through
each update.  For the affine model the data-loss Hessian is constant, while
`mlp_loss_hvp` supplies the exact joint `(theta,l2)` product.  The validation
contraction is therefore

```text
H_outer(p) d = J_validation(theta) theta_{p,d}
              + theta_p^T H_validation(theta) theta_d.
```

No finite differences or third network derivatives are used by the production
path.  `hvp(parameters, direction, product, status)` returns
`FORTNUM_NOT_IMPLEMENTED` for nonlinear networks or nonconstant schedules,
because those cases need respectively third network derivatives or schedule
rate second products.  CUDA remains an explicit refusal until a resident
trajectory kernel is available.

The regression test checks the exact CPU product against an independent central
finite difference of the complete packed reverse gradient, checks Hessian
symmetry with two directions, and exercises the FortOpt callback.  The release
workload is `fortml_bench_mlp_constant_schedule_hvp`; the companion benchmark
uses an independent NumPy affine recurrence and records value, gradient, JVP,
and HVP arrays before retaining timings.
