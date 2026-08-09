# RMSprop trajectory hypergradient HVP

`mlp_rmsprop_hypergradient_objective_t` models a fixed full-batch RMSprop
trajectory and exposes value, gradient, JVP, VJP, and HVP products over

```text
[ log(learning_rate), log(l2), decay, log(epsilon), momentum ]
```

The `centered` option selects the RMSprop variance recurrence when the
objective is initialized. It is a discrete model choice. The packed vector
contains only continuous coordinates. Both centered and uncentered branches
carry first and mixed second state sensitivities.

## Affine outer HVP

For one affine linear layer, the MSE data Hessian is independent of the model
parameters. The `hvp` binding propagates a directional derivative of each
packed hypergradient through the square-average, centered-average, and
momentum states. The validation product is

```text
H(theta_outer) * direction
```

where the direction is a five-component packed vector. The implementation uses
`mlp_loss_hvp` for the affine network block and carries the L2 mixed terms
analytically. FortOpt receives the same value-gradient callback through
`fortopt`, so L-BFGS-B and direct products share one trajectory definition.

The affine restriction is a derivative contract. A nonlinear or multilayer
network needs a third network derivative in the outer HVP. Such a request
returns `FORTNUM_NOT_IMPLEMENTED` with the message
`nonlinear network needs third derivatives`.

The variance square-root follows the positive-variance branch used by the
first-order product. A negative variance state is reported as
`FORTNUM_DOMAIN_ERROR`. The centered recurrence does not silently continue
through an invalid state.

## Verification

`test_mlp_rmsprop_hypergradient` checks centered and uncentered affine HVPs
against an independent central-difference gradient oracle. It also checks the
FortOpt context adapter and the typed nonlinear refusal. The release workload
is `fortml_bench_rmsprop_hypergradient`. The benchmark report and CSV are in
the FortML-bench repository at
`results/RMSPROP_HYPERGRADIENT_HVP.md` and
`results/rmsprop_hypergradient_hvp.csv`.
