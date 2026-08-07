# Physics-informed and structure-preserving models

This page separates the scientific-ML research target from the small
Hamiltonian prototype that is currently released. It is an API and evidence
contract, not a claim that PINNs, symplectic GPs, or GP-initialized finite
networks are complete.

## Current implementation

`fortml_hamiltonian_mlp` provides `hamiltonian_mlp_t` for a separable scalar
Hamiltonian,

```text
H(q,p) = V(q) + T(p),       f(q,p) = ( dT/dp, -dV/dq ).
```

The two scalar MLPs have a deterministic packed parameter vector. The public
products are `energy`, `energy_gradient`, `vector_field`, `energy_jvp`,
`energy_vjp`, `vector_field_jvp`, and an explicit velocity-Verlet (`leapfrog`)
step. The implementation refuses malformed layer shapes, non-finite states,
and non-finite directions. The independent test
[`test_hamiltonian_mlp.f90`](../test/test_hamiltonian_mlp.f90) checks the value
products against central differences, the VJP dot-product identity, and the
finite-difference Jacobian's symplectic-form defect and reversibility.

This is deliberately narrower than a general HNN or LNN: the current model is
separable, has canonical coordinates, and does not learn a Poisson tensor or
an implicit integrator. `linear_autoencoder_t` is likewise only the exact
tied, centered PCA reconstruction seam; it is not a nonlinear or physics
autoencoder initializer.

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
  independent skew-symmetry and Jacobi checks; a general Hamiltonian requires
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
These references motivate testable contracts; they do not by themselves
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
refusal or an `unavailable` benchmark row; no hidden host fallback or relabeled
CPU timing is permitted.

The benchmark evidence in
[`../fortml-bench/results/PHYSICS_MODELS.md`](../../fortml-bench/results/PHYSICS_MODELS.md)
records this distinction and includes a small independent NumPy symplectic
map oracle alongside the current CPU product gate.

