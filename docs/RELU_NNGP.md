# ReLU NNGP covariance

`fortml_relu_nngp` evaluates the exact infinite-width covariance of a dense,
fully connected ReLU hidden stack with independent Gaussian priors:

```text
W_l,ij ~ N(0, sigma_w^2 / fan_in)
b_l,j  ~ N(0, sigma_b^2)
```

Construct `relu_nngp_t`, call `configure(input_dimension, hidden_layer_count,
status[, weight_variance, bias_variance])`, then call
`covariance(x_left, x_right, value, status)`. Inputs are row-major batches.
The input covariance is `x_left x_right^T / input_dimension`; each hidden
layer applies the analytic ReLU arc-cosine recurrence. `weight_variance=2` and
`bias_variance=0` are the He-prior defaults.

The metadata sets `exact_infinite_width=.true.` because the returned kernel is
the analytic GP limit. It also sets
`finite_mlp_weight_map_supported=.false.`: this module does not claim that it
can deterministically assign finite MLP weights realizing that covariance.
`covariance_cuda` is an explicit `FORTNUM_NOT_IMPLEMENTED` refusal and leaves
its output unchanged; no host computation is relabeled as GPU work.

The contract currently covers ReLU only. Finite-width ensemble calibration,
other activations, output-layer priors, posterior solves, sampled weight maps,
and a resident CUDA kernel are separate work.
