# Weighted validation for the SGD trajectory objective

`fortml_mlp_sgd_momentum_hypergradient` accepts an optional
`validation_weight(:)` vector on `initialize` and on
`mlp_optimize_sgd_momentum_hyperparameters`.  The vector has one finite
non-negative entry per validation row and must have positive total mass.  The
outer validation loss is the weighted mean, matching the weighted-MSE
reduction used by the shared loss facade.  Zero-weight rows are therefore
valid, while an all-zero vector is rejected transactionally.

The weighted validation measure is held in the objective state.  Exact
`value_gradient`, scalar `jvp`, and scalar `vjp` products propagate the same
measure through the full fixed SGD momentum/Nesterov trajectory; no
finite-difference code is used in the implementation.  The objective remains
compatible with the existing FortOpt L-BFGS-B adapter.

The affine one-layer outer HVP is certified for a uniform validation measure
(including the default unweighted path).  A non-uniform validation measure
returns `FORTNUM_NOT_IMPLEMENTED` from `hvp` rather than silently applying an
uncertified residual contraction.  CUDA and stochastic trajectories retain the
existing typed refusal contract.

`test_mlp_weighted_validation_hypergradient` checks central-difference
hypergradients, JVP/VJP adjoint behavior, the certified uniform HVP, invalid
weight handling, and the non-uniform/CUDA refusal boundaries.
