# Weighted RMSprop trajectory hypergradients

`mlp_rmsprop_hypergradient_objective_t` accepts optional training and
validation row weights. The fixed full-batch trajectory differentiates the
same five coordinates as the unweighted contract:

```text
[ log(learning_rate), log(l2), decay, log(epsilon), momentum ]
```

Training weights enter every MSE gradient and HVP before the square-average,
centered-average, and momentum updates. Validation weights enter the outer
value, gradient, and affine HVP. Each weighted mean divides by positive weight
mass. L2 remains a single parameter regularizer and is not scaled by row mass.

`initialize` and `mlp_optimize_rmsprop_hyperparameters` take `train_weight` and
`validation_weight` after the status argument. A weight vector must match its
row count, contain finite nonnegative values, and have positive total mass.
Rejected weights leave the adapter uninitialized and preserve the model
parameters. Metadata records whether each dataset is weighted and its accepted
mass.

The CPU recurrence supports value-gradient, JVP, VJP, affine outer HVP, and the
FortOpt L-BFGS-B callback. Nonlinear outer HVPs retain the existing typed
third-derivative refusal. CUDA trajectory requests return
`FORTNUM_NOT_IMPLEMENTED` until the model, weighted reductions, optimizer
state, and their sensitivities remain resident together.

`test_mlp_rmsprop_weighted_hypergradient` compares all five derivatives and the
outer HVP against a separate scalar affine RMSprop recurrence. It checks the
FortOpt callback, bounded weighted optimization, metadata, malformed-weight
rollback, and CUDA refusal. `fortml_bench_rmsprop_weighted_hypergradient`
provides release values and CPU timings for the companion benchmark harness.
