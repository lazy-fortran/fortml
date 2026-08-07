# External parity reference

The parity target follows the public feature surfaces of the reference
packages. The links below are the source index used when ROADMAP items are
split into implementation slices. A row describes a target contract. It does
not imply that FortML already implements the contract.

## scikit-learn

The [user guide](https://scikit-learn.org/stable/user_guide) covers linear and
kernel models, discriminant analysis, support-vector machines, trees and
ensembles, calibration, clustering, decomposition, preprocessing, pipelines,
model selection, inspection, and classification and regression metrics. The
[metrics API](https://scikit-learn.org/stable/api/sklearn.metrics.html) and
[Pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline)
define the metric names, scorer construction, metadata routing, and leakage
boundaries that FortML must reproduce where the numerical objective matches.

The remaining FortML workflow gaps are sparse and categorical views,
cross-validation and estimator cloning, calibration-aware model selection,
partial-fit and online estimators, kernel and one-class SVMs, clustering and
mixture families, manifold methods, complete missing-value policies, and
serialization and serving contracts.

## PyTorch

The [PyTorch documentation](https://docs.pytorch.org/docs/stable/) defines
tensor and module state, data loading, automatic mixed precision, optimizers,
compiled execution, distributed training, checkpointing, and operator-level
autodiff. [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
adds graph capture and guarded compilation. The
[distributed package](https://docs.pytorch.org/docs/stable/distributed) adds
multi-process collectives and data-parallel training.

FortML's production trainer target therefore includes parameter and buffer
trees, train and evaluation modes, deterministic data loaders, gradient
accumulation, clipping, mixed precision with loss scaling, compiled static
graphs, distributed reduction, checkpoint schemas, and optimizer state
resumption. Each device path must expose residency and transfer accounting.

## JAX

The [JAX documentation](https://docs.jax.dev/) defines composable `jit`,
`vmap`, forward and reverse autodiff, pytrees, explicit device placement, and
functional state updates. Current multi-device guidance uses meshes,
`NamedSharding`, and `shard_map`. The
[shard_map guide](https://docs.jax.dev/en/latest/notebooks/shard_map.html)
requires explicit input and output partition specifications and collectives.

FortML's corresponding target is a static expression and parameter graph with
JVP, VJP, HVP, batching, donation or ownership rules, explicit sharding, and
collective reduction semantics. A host callback or an unrecorded transfer is a
capability refusal, not a GPU result.

## GPyTorch and GPflow

The [GPyTorch documentation](https://docs.gpytorch.ai/en/stable/) covers exact
and approximate GP models, lazy linear operators, kernels, likelihoods,
variational strategies, inducing points, stochastic ELBO objectives, natural
gradients, batch shapes, priors, constraints, prediction strategy, and
matrix-free inference. Its
[likelihood API](https://docs.gpytorch.ai/en/stable/likelihoods.html) separates
latent distributions from observed-data mappings, while the
[variational API](https://docs.gpytorch.ai/en/stable/variational.html)
specifies approximate posterior and ELBO state.

The FortML GP target adds mean functions, ARD and batch or multitask shapes,
all supported likelihood links, derivative and operator observations, exact
and variational parameter products, implicit solve derivatives, sparse and
SKI products, train-state serialization, and resident CUDA covariance and
factorization. The current capability matrix records the smaller CPU surface
and every typed refusal.

## XGBoost, LightGBM, Flux, and Lux

The XGBoost target includes histogram and quantile sketch construction,
missing-value default directions, monotone and interaction constraints,
ranking objectives, categorical partitions, DART, early stopping, staged
contributions, model dumps, and distributed workers. LightGBM adds leaf-wise
growth, GOSS, EFB, categorical handling, and distributed histogram reduction.
Flux and Lux add composable Julia module trees, parameter and buffer selection,
training state, callbacks, and GPU array execution. FortML records these as
separate contracts because tree split selection, autodiff module execution,
and fixed no-autodiff GPU kernels have different derivative and residency
requirements.

## Acceptance rules

Every completed parity item needs a public API, an independent behavioral
oracle, refusal tests for unsupported inputs or devices, documentation, and a
benchmark row with source and toolchain provenance. CPU correctness, resident
GPU correctness, and transfer-inclusive GPU measurements are separate evidence
classes. A benchmark cannot promote a typed refusal to a CPU or GPU timing.
