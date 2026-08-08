# Sparse variational GP products

`sparse_gp_t` now exposes an exact CPU product contract for its scalar Gaussian
ELBO. The packed variational vector is

```text
[ mean(1:n_inducing), log(L(1,1)), L(2,1), L(3,1), ..., log(L(n,n)) ]
```

where `L` is the lower Cholesky factor of `q(u)` and entries are ordered by
columns. `parameter_count`, `parameters`, and `set_parameters` own this
coordinate transform. `elbo_gradient`, `elbo_jvp`, and the scalar-cotangent
`elbo_vjp` differentiate the closed-form Gaussian expected likelihood and
analytic `KL(q(u)||N(0,K_uu))`; kernel, inducing-location, and observation-noise
coordinates remain fixed in this slice. `elbo_kernel_parameter_jvp` and
`elbo_kernel_parameter_vjp` provide the complementary fixed-state products for
the packed `kernel%parameters()` vector: the inducing solve is differentiated
explicitly while the variational mean/factor, inducing locations, and noise
variance remain fixed. These products are exact on CPU for every kernel whose
`matrix_jvp`/`parameter_vjp` methods are available, including composed kernels.

The independent `test_sparse_gp` oracle checks every packed gradient coordinate
against central differences, a directional JVP against a central directional
difference, and the VJP/JVP dot-product identity. `elbo_device`,
`elbo_gradient_device`, and `elbo_jvp_device` dispatch CPU directly and return a
typed `FORTNUM_NOT_IMPLEMENTED` refusal for CUDA until the inducing solve and
reductions have resident kernels. The kernel-parameter product dispatchers use
the same CPU path and typed CUDA refusal.
