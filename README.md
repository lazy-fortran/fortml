# fortml

`fortml` is a differentiable machine-learning library for modern Fortran. It
builds on `fortnum` for numerical kernels and on `fortad` for generated
derivative code. Models expose value, JVP, and VJP products where the
mathematics has been verified.

The implemented baselines are multi-output ridge/ordinary least-squares
regression, an explicit-parameter MLP, and a multi-output exact Gaussian
process with composable kernels. The MLP and GP expose value/JVP/VJP products.
The planned model families are derivative-aware Gaussian
processes, scalable<!-- slop-ok --> and multi-output GPs, variational autoencoders, and deep
recurrent networks. Bayesian global optimization is outside this repository's
current scope.

See [ROADMAP.md](ROADMAP.md) for the gated work order and the local
`.provenance/` directory for the ignored literature and upstream source corpus.
