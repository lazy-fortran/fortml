# Multi-output GP products

`multi_output_gp_t` is an intrinsic-coregionalization exact GP with
`B = W W^T + diag(independent)`.  Training targets and posterior means use
`(sample, output)` order.  Product routines keep an internal output-major
stacking (`output * n_samples + sample`) so the Kronecker layout is explicit
and independent of Fortran's column-major flattening of the public arrays.

The packed fitted-model coordinates returned by `parameters()` are:

```
[ kernel%parameters(), log(noise_variance),
  W(1,1:rank), W(2,1:rank), ..., W(n_outputs,1:rank),
  independent(1:n_outputs) ]
```

`parameter_count()` reports this length.  `predict_parameter_jvp` and
`predict_parameter_vjp` differentiate the fitted posterior mean through both
the cross-covariance and the differentiated Cholesky solve.  Kernel products
are delegated to `kernel_t%matrix_jvp` and `kernel_t%parameter_vjp`; the
coregionalization and log-noise blocks are assembled analytically.  These are
fixed-data products: training inputs, targets, and output count are held
fixed, while the fitted solve state is differentiated exactly.

`predict_input_jvp(query,direction,mean,mean_dot,status)` and
`predict_input_vjp(query,mean_bar,query_bar,status)` hold the fitted model
state fixed and differentiate only query locations using
`kernel_t%input_derivatives`.  The VJP is the adjoint of the JVP under the
same output-major stacking.  No production finite differences are used.

For independent query sets, `predict_batch(query,mean,status)` accepts
`query(batch,query,feature)` and returns `mean(batch,query,output)`. The
`predict_batch_input_jvp` and `predict_batch_input_vjp` methods apply the same
fixed-fit products member by member while preserving that shape contract.
`test_multi_output_gp_batch` checks the batched mean against an independent
dense RBF oracle, checks the JVP by central differences and the VJP by scalar
duality, and exercises malformed-shape and nonfinite-input refusals.

The four existing `*_device` wrappers and the three batch `*_device` wrappers
accept CPU execution and return
`FORTNUM_NOT_IMPLEMENTED` for CUDA until resident coregionalized covariance,
factorization, and derivative kernels are available.  They never silently
copy to a host fallback.  `test_multi_output_gp_products` checks query and
parameter JVPs against independent central-difference refits, checks both
VJP adjoint identities, and checks the typed CUDA refusals.
