# Column basis pipeline device contract

`column_basis_pipeline_t` is the differentiable, column-selecting feature
union. The host products already provide exact transform, JVP, VJP, and HVP
operations. The typed device entry points make the execution boundary explicit:

```fortran
call cpu%select(FORTML_DEVICE_CPU, status)
call pipeline%transform_device(cpu, x, features, status)
call pipeline%jvp_device(cpu, x, theta_dot, x_dot, features, features_dot, status)
call pipeline%vjp_device(cpu, x, u, theta_bar, x_bar, status)
call pipeline%hvp_device(cpu, x, u, theta_dot, x_dot, theta_hvp, x_hvp, status)
```

CPU dispatch delegates to the corresponding exact host operation. CUDA is
currently a typed refusal (`FORTNUM_NOT_IMPLEMENTED`), and every refusal leaves
all output arrays unchanged; there is no hidden host fallback or implicit data
transfer. `device_supported(FORTML_DEVICE_CPU)` is true only after `fit`, while
CUDA and invalid device kinds are false. A future resident basis executor can
implement the same methods without changing the feature or derivative API.

The independent `test_column_pipeline` device cases compare every CPU product
against its host result and seed CUDA outputs with sentinels before checking
the refusal contract.
