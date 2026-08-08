# Derivative capability matrix

This is the canonical schema for recording what a FortML object can compute.
It is deliberately separate from an estimator name: a classifier, kernel, or
trainer is not considered differentiable merely because one of its methods has
the word `gradient` in its name.

## One row per public product

Every public estimator, transform, basis, kernel, likelihood, objective,
optimizer trajectory, and device plan should be represented by a row with these
fields:

| Field | Meaning |
| --- | --- |
| `owner` / `object` | Module and concrete type that owns the state. |
| `product` | `value`, `input_jvp`, `input_vjp`, `input_hvp`, `parameter_jvp`, `parameter_vjp`, `parameter_hvp`, `hyperparameter_jvp`, `hyperparameter_vjp`, or `hyperparameter_hvp`. |
| `state_mode` | `unfitted`, `fixed_fit`, `trajectory`, `implicit_optimum`, or `stochastic_path`. |
| `mode` | `analytic`, `fortsym`, `fortad`, `finite_difference_oracle`, or `refused`. A finite-difference oracle is evidence, never the implementation mode. |
| `blocks` | Named parameter and input blocks covered by the product, including offsets and transforms. |
| `smoothness` | Smooth domain, exact kink, active-set, split-routing, or discrete boundary. |
| `device` | CPU, resident CUDA, OpenACC, transfer-inclusive, or typed refusal. |
| `status` | The exact refusal/status code and the condition that produces it. |
| `oracle` | Independent analytic, dense, finite-difference, adjoint, or pinned external reference. |
| `test` / `benchmark` | Behavioral test and correctness-gated benchmark record. |
| `provenance` | FortML, dependency, compiler, generator, and benchmark revisions. |

An absent row means “not audited”; it does not mean zero. A `refused` row is
valid evidence only when the refusal is deterministic, non-mutating, and has a
test plus a benchmark capability record. CPU success does not imply device
success, and a fixed-state derivative does not imply a derivative through fit,
validation, RNG, early stopping, or an optimizer trajectory.

## Canonical product rules

1. JVPs are checked against a convergence-tested directional perturbation.
2. VJPs satisfy the scalar tangent/cotangent dot-product identity.
3. HVPs are checked against an independently differentiated gradient or a
   symmetric dense Hessian-vector reference.
4. Hyperparameter products use the same transformed parameter registry and
   callback consumed by FortOpt L-BFGS-B. An optimizer may not silently omit a
   registered block.
5. Nonsmooth split, routing, sign, max-state, clipping, and exact-kink
   boundaries return a named refusal or a fixed-active-set contract.
6. A CUDA/OpenACC row is resident only when model, optimizer, batch, workspace,
   and derivative state remain on the device. Otherwise it is a typed refusal
   or a separately labelled transfer-inclusive call.
7. Every row is regenerated when a source, dependency, compiler, generated
   FortSym/FortAD leaf, or benchmark harness revision changes.

## Example

The following is the minimum record for a fixed-input multiclass MLP product:

```text
owner=fortml_mlp_classifier_t
product=parameter_jvp
state_mode=fixed_fit
mode=analytic
blocks=layer_1.weight,layer_1.bias,layer_2.weight,layer_2.bias
smoothness=softmax probabilities; labels and argmax are discrete
device=CPU success; resident CUDA FORTNUM_NOT_IMPLEMENTED
oracle=independent tanh/softmax replay + central difference
test=test_mlp_classifier_parameter_products
benchmark=results/mlp_classifier_parameter_products.csv
```

The same schema applies to GP derivative observations, basis-pipeline feature
maps, XGBoost fixed-tree contributions, and neural optimizer trajectories. The
implementation gap remains open until the rows exist for every declared public
surface; this document defines the acceptance format rather than claiming that
the matrix is already complete.
