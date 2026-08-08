# Change-point kernels

`make_change_point_kernel` blends two child kernels across one input feature
while preserving the positive-semidefinite covariance construction.

```fortran
use fortml_kernels, only: kernel_t, make_change_point_kernel, make_rbf_kernel, &
    make_constant_kernel

type(kernel_t) :: left, right, kernel
type(fortnum_status_t) :: status

left = make_rbf_kernel(2, 1.4_dp, 0.6_dp, status)
right = make_constant_kernel(2, 0.4_dp, status)
kernel = make_change_point_kernel(left, right, 2, 0.15_dp, 0.7_dp, status)
```

For feature `d`, the gate is

```text
s(x) = (1 + tanh((x_d - center) / width)) / 2
```

and the covariance is

```text
k(x,x') = s(x)s(x') k_left(x,x')
        + (1-s(x))(1-s(x')) k_right(x,x')
```

`width` must be positive. The kernel stores the unconstrained parameters
`log(width)` and `center` after all child parameters. The `matrix`,
`matrix_jvp`, `parameter_vjp`, and `parameter_hvp` methods expose analytic
products for exact-GP likelihood and optimizer derivatives. `input_derivatives`
includes the gate derivatives and the mixed input Hessian, so derivative
observations use the same API as other smooth kernels.

The constructor checks child dimensions and the selected feature. CUDA requests
continue to follow FortML's typed device contract. The CPU exact-GP path is
implemented. A resident CUDA change-point covariance kernel is still an open
roadmap item.
