# Unfactored Adafactor trajectory hypergradients

`fortml_mlp_adafactor_hypergradient` provides an exact deterministic,
fixed-full-batch outer objective for the vector Adafactor recurrence already
used by `fortml_trainer`. Its packed search vector is

```text
[ log(learning_rate), log(l2), decay, log(epsilon), log(clip_threshold) ]
```

The inner state is reset to the MLP parameters captured by `initialize`, then
the configured number of updates is applied to the training rows. The scalar
objective is unregularized validation MSE. The regularized MLP gradient feeds
the unfactored second-moment state, followed by RMS clipping and the
epsilon-stabilized update used by `fortml_adafactor`.

`value_gradient` and `jvp` propagate exact tangents through the parameter,
second-moment, RMS, clip-scale, and denominator states. The `log(l2)` product
uses the analytic MLP HVP, while `decay`, epsilon, and clip-threshold products
differentiate their scalar transforms. `vjp` is the scalar adjoint identity,
and `fortopt` adapts the objective directly to FortOpt's L-BFGS-B callback.
`mlp_optimize_adafactor_hyperparameters` supplies explicit bounds and reports
both packed and physical values.

The derivative contract is intentionally explicit:

- `relative_step` and `scale_parameter` are fixed discrete branches and return
  `FORTNUM_NOT_IMPLEMENTED` until their state derivatives are added.
- A trajectory exactly on `update_rms == clip_threshold` returns
  `FORTNUM_NOT_IMPLEMENTED`; differentiating across that active-set kink would
  be unsound.
- CUDA and resident-device trajectories return a typed refusal. No host
  fallback is presented as GPU evidence.

The independent `test_mlp_adafactor_hypergradient` fixture checks all five
packed components against central differences, a directional JVP, the scalar
VJP identity, the FortOpt context adapter, and the CUDA/discrete-branch
refusals. The products are analytic; finite differences are used only by that
behavioral oracle.
