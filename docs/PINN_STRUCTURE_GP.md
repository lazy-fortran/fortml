# PINN structure-aware finite-feature GP initialization

`fortml_pinn_structure_gp` provides a bounded warm start for a fixed-depth
MLP used by `physics_objective_t`. It freezes the hidden feature map, solves
the finite-feature kernel-ridge output layer, and records the four named
objective terms in `[data, residual, boundary, conservation]` order.

The initializer is a deterministic finite-width posterior-mean map, not an
NNGP or exact infinite-width equivalence. It validates the hidden parameter
snapshot and RMS scale before every apply, prediction, variance, or
regularization-JVP query; changed topology or hidden state returns a typed
domain error. CPU products are exact for the frozen feature map. CUDA apply,
prediction, JVP, and variance calls return `FORTNUM_NOT_IMPLEMENTED` without
mutating host outputs.

The independent manufactured-PDE test `test_pinn_structure_gp` checks the
dense normal-equation coefficients, prediction, named objective diagnostics,
hidden-state transaction, and typed CUDA boundary. The release application is
`fortml_bench_pinn_structure_gp`; its NumPy companion records the same oracle
and capability row in `fortml-bench/results/PINN_STRUCTURE_GP.md`.
