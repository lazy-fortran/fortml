# Physics-informed and structure-preserving models

This page separates the scientific-ML research target from the small
Hamiltonian prototype that is currently released. It is an API and evidence
contract, not a claim that PINNs, symplectic GPs, or GP-initialized finite
networks are complete.

## Current implementation

`fortml_hamiltonian_mlp` provides `hamiltonian_mlp_t` for separable and general
scalar Hamiltonians. The original initializer stores two scalar networks,

```text
H(q,p) = V(q) + T(p),       f(q,p) = ( dT/dp, -dV/dq ).
```

The two scalar MLPs have a deterministic packed parameter vector. The public
products are `energy`, `energy_gradient`, `vector_field`, `energy_jvp`,
`energy_vjp`, `vector_field_jvp`, and an explicit velocity-Verlet (`leapfrog`)
step. `initialize_general` selects one scalar MLP over the full `[q,p]` state;
the same products then differentiate a nonseparable Hamiltonian without a
finite-difference fallback. The explicit leapfrog map is refused with
`FORTNUM_NOT_IMPLEMENTED` in general mode because it is a split integrator;
an implicit symplectic method is a separate contract. Both modes refuse
malformed layer shapes, non-finite states, and non-finite directions. The independent test
[`test_hamiltonian_mlp.f90`](../test/test_hamiltonian_mlp.f90) checks the value
products against central differences, the VJP dot-product identity, general
state/parameter JVPs, and the separable finite-difference Jacobian's
symplectic-form defect and reversibility.

The Hamiltonian model has canonical coordinates and does not learn a Poisson
tensor or an implicit integrator. `linear_autoencoder_t` is likewise only the exact
tied, centered PCA reconstruction seam. It is not a nonlinear or physics
autoencoder initializer.

`mlp_t%initialize_from_pca` exposes the same fixed-width reconstruction map as
a two-layer linear MLP, including the center and optional PCA whitening scales.
It is a checked finite linear/PCA optimum and leaves an existing model
unchanged when the PCA is unfitted or malformed. It does not claim an NNGP,
NTK, GP-posterior, physics-consistent, symplectic, or Hamiltonian network
equivalence; those finite/infinite-width and structure-preserving mappings
remain separate research contracts.

`fortml_physics_objective` now provides the first composable residual seam:
`physics_constraint_t` owns a positive reduction weight and caller-supplied
residual/JVP/VJP callbacks, while `physics_objective_t` sums named data,
residual, boundary, and conservation slots and adapts value/gradient to
FortOpt. `term_values(theta, values, status)` exposes the four normalized
weighted contributions in the fixed order `[data, residual, boundary,
conservation]`; inactive slots are zero and the entries sum to the objective
value. This makes residual balancing and conservation monitoring observable
without coupling callers to private constraint storage. A constraint may now
also register `physics_residual_hvp_proc`, a reverse-over-forward callback
that receives normalized residual and residual-JVP cotangents and returns the
exact weighted least-squares HVP. This is the derivative contract needed by
FortOpt L-BFGS-B for nonlinear PINN, HNN, and symplectic residual providers; it
does not form a Jacobian or Hessian and has no finite-difference fallback.
Providers without the optional callback retain a typed
`FORTNUM_NOT_IMPLEMENTED` HVP refusal. The independent affine and nonlinear
oracles are in [`test_physics_objective.f90`](../test/test_physics_objective.f90).
Coordinate/time metadata, collocation samplers, PINN/GP adapters, and
symplectic terms remain future work. The diagnostic itself is CPU/device
agnostic and does not introduce a host fallback: a resident adapter remains
responsible for its residual callback and products.

`fortml_pinn` adds `pinn_training_adapter_t` as the training-facing facade for
that composition. `initialize(objective,status[,device_kind])` accepts an
initialized `physics_objective_t` and forwards `value`, `gradient`,
`value_gradient`, `jvp`, `vjp`, `hvp`, and `term_values` without copying model
or collocation state out of callback-owned contexts. `fit_lbfgsb` adapts the
same objective to FortOpt's bounded L-BFGS-B implementation; the caller owns
the packed parameter vector and bounds. The manufactured-solution
[`test_pinn.f90`](../test/test_pinn.f90) checks all four named terms, central
finite-difference JVP/HVP oracles, scalar VJP, the bounded fit, and malformed
shapes. CPU is the only supported device in this slice. A CUDA initialization
or selection request returns `FORTNUM_NOT_IMPLEMENTED` and does not execute a
host fallback.

## Research contracts

The next APIs should compose with the existing parameter registry and objective
products rather than introduce a second training stack:

* `physics_constraint_t` should expose residual value/JVP/VJP/HVP, coordinate
  and time layouts, units, boundary masks, and named reduction weights.
* PINN and physics-informed GP objectives should keep data, PDE/ODE residual,
  initial/boundary, conservation, and symplectic terms separately observable.
  Collocation and trajectory samplers must retain their seed and emitted
  points.
* General HNN/LNN, SympNet, and symplectic recurrent maps need a declared
  structure certificate. A learned Poisson tensor is accepted only with
  independent skew-symmetry and Jacobi checks. A general Hamiltonian requires
  an applicable implicit symplectic method or a typed refusal.
* Physics-consistent and symplectic GPs should register linear differential
  operators and adjoints, then expose mixed value/derivative covariance
  products. Ghosttasking/Monge-GP and the FortML-author symplectic-GP results
  are experimental reference fixtures until public equations, data, and
  implementations are pinned.
* NNGP/NTK, GP-posterior, PCA, and basis initializers must state whether they
  preserve a mean, approximate a covariance, or reproduce a linear projection.
  A finite-width network is not silently advertised as exactly equal to its
  infinite-width GP.

The literature motivating these contracts includes
[Hamiltonian Neural Networks](https://papers.nips.cc/paper/9672-hamiltonian-neural-networks.pdf),
[Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630),
[SympNets](https://arxiv.org/abs/2001.03750),
[Physics-informed neural networks](https://doi.org/10.1016/j.jcp.2018.10.045),
the PIML review by [Karniadakis et al.](https://doi.org/10.1038/s42254-021-00314-5),
[Symplectic Gaussian Process Regression of Hamiltonian Flow Maps](https://arxiv.org/abs/2009.05569),
[Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165),
and the linear autoencoder/PCA optimum of
[Baldi and Hornik](https://doi.org/10.1016/0893-6080(89)90014-2).
These references motivate testable contracts. They do not by themselves
constitute FortML implementation evidence.

## Device contract

The CPU implementation is the behavioral reference. The current Hamiltonian
MLP has no resident model or optimizer CUDA plan and therefore has no
end-to-end GPU claim. OpenACC may be used only when the complete residual,
derivative, and integration graph remains resident and agrees with the CPU
oracle. Fixed no-autodiff reductions or map kernels may use native CUDA when a
FortSym-derived or hand-derived expression has an independent oracle. A
FortAD-bearing path must stay on the CPU/reference graph until its complete
device derivative graph exists. Missing coverage is reported as a typed
refusal or an `unavailable` benchmark row. No hidden host fallback or relabeled
CPU timing is permitted.

The benchmark evidence in
[`../fortml-bench/results/PHYSICS_MODELS.md`](../../fortml-bench/results/PHYSICS_MODELS.md)
records this distinction and includes a small independent NumPy symplectic
map oracle alongside the current CPU product gate.
