# fortml

`fortml` is a differentiable machine-learning library for modern Fortran. It
builds on `fortnum` for numerical kernels and on `fortad` for generated
derivative code. Models expose value, JVP, and VJP products where the
mathematics has been verified.

The implemented baselines are multi-output ridge/ordinary least-squares
regression, an explicit-parameter MLP, a Bayesian neural network with a reparameterized
Gaussian variational posterior and seeded Monte Carlo ELBO
value/JVP/VJP/HVP products, and a multi-output exact Gaussian
process with composable kernels. The MLP and GP expose value/JVP/VJP products.
The RBF derivative-GP pilot now supports mixed function-value and first-input
derivative observations and predictions, with symbolic and finite-difference
checks. The structured GP operator accepts separable tensor-grid covariance
factors and reuses the matrix-free CG contract; the Toeplitz GP operator wraps
the cached FFT grid product for one-dimensional covariance structures. The
compact-support sparse GP operator consumes `fortsparse` triplets, retains a
CSR view for row-owned products, and has a resident OpenACC path. The
generic kernel operator lowers leaf RBF expressions to the same fused,
matrix-free OpenACC/native-CUDA product and exposes explicit device-data
lifetime hooks. Built-in sum/product trees are flattened to a static postfix
program for the same device path; user-supplied formulas remain a separate
lowering milestone. The
planned model families are
scalable<!-- slop-ok --> and multi-output GPs,
variational autoencoders, and deep recurrent networks. Bayesian global
optimization is outside this repository's current scope.

See [ROADMAP.md](ROADMAP.md) for the gated work order and the local
`.provenance/` directory for the ignored literature and upstream source corpus.
