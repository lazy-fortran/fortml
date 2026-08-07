# Spectral-mixture Gaussian-process kernel

`make_spectral_mixture_kernel(input_dim, num_mixtures, weights, means,
scales, status)` adds a stationary spectral-mixture leaf to the exact GP
kernel tree.  It follows the GPyTorch convention

```
k(tau) = sum_q w_q prod_d exp(-2*pi^2*tau_d^2*s_qd^2)
                         * cos(2*pi*tau_d*mu_qd)
```

where `tau = x1 - x2`, `weights` and positive frequency-standard-deviation
`scales` are strictly positive, and `means` are signed frequencies.  `means`
and `scales` have shape
`(num_mixtures, input_dim)`.  The constructor validates dimensions, positivity,
and finiteness before allocating state.

The packed parameter contract is compositional and stable:

```
[log_weight(q), log_scale(q,1:input_dim), mean(q,1:input_dim)] for q=1..Q
```

Thus `parameter_count()` is `Q*(1 + 2*input_dim)`, and the leaf can be used
unchanged in `kernel_add` and `kernel_multiply` trees.  Positive hyperparameters
remain unconstrained in FortOpt's log coordinates while frequencies retain their
signed physical coordinates.

The CPU reference path supplies scalar and dense-matrix values, exact input
JVP/VJP-compatible gradients and mixed input Hessians, parameter matrix JVPs,
VJPs, and directional parameter HVPs.  The products are closed form (no
finite-difference fallback): the independent test
`test_gp_spectral_mixture_kernel` checks values against the formula, input
derivatives, matrix JVPs, the VJP adjoint identity, and HVP finite differences;
it also fits and predicts an exact GP with the new leaf.  Exact GP regression
therefore inherits its ordinary likelihood and prediction hyperparameter
products through the existing `kernel_t` contract.

No resident OpenACC or CUDA spectral-mixture kernel is linked yet.  Device
selection must report the existing typed `FORTNUM_NOT_IMPLEMENTED` refusal;
there is no silent host fallback.  A resident implementation should preserve
the packed layout and use an independently assembled CPU oracle before being
promoted in the device capability table.
