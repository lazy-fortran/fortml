# Scheduled AdamW trajectory hypergradients

`fortml_mlp_adamw_schedule_hypergradient` differentiates a deterministic,
fixed full-batch AdamW trajectory with a typed stateless learning-rate
schedule. The packed outer coordinates are

```text
[ log(base_rate), log(l2), log(weight_decay), logit(beta1), logit(beta2),
  log(epsilon), logit(min_rate_fraction), logit(decay_factor) ]
```

Constant, cosine, warmup-cosine, and exponential-decay schedules are supported;
the schedule kind and integer update counts are fixed. The value/gradient, JVP,
scalar VJP, and FortOpt L-BFGS-B adapter propagate parameter, moment,
bias-correction, decoupled-decay, and schedule-coordinate sensitivities
analytically. No inner finite-difference training run is used.

The CPU reference path is explicit. CUDA and lower-precision requests return
`FORTNUM_NOT_IMPLEMENTED` until resident MLP, schedule, and optimizer state
contracts exist. The outer HVP returns the same typed refusal because a general
nonlinear network needs third derivatives; zero second-moment derivatives are
also refused rather than assigned a hidden subgradient. The independent
`test_mlp_adamw_schedule_hypergradient` fixture checks all eight coordinates by
central differences, a directional JVP, scalar-VJP adjointness, cosine and
exponential branches, FortOpt convergence, and device/optimizer refusals.

The release workload and independent NumPy replay are recorded in
`fortml-bench/results/MLP_ADAMW_SCHEDULE_HYPERGRADIENT.md`.
