# FortML parity matrix

This matrix is the acceptance contract for estimator, Gaussian-process, and
neural-network parity. A row is complete only when the implementation, an
independent behavioral oracle, a documented derivative boundary, a device
record, and a companion benchmark all agree. A typed refusal is evidence for a
missing backend. It is not a CPU timing reported as GPU support.

## Estimator families

| Family | Current FortML surface | Required closure |
| --- | --- | --- |
| Linear and GLM | Linear, ridge, elastic-net/lasso, weighted SVR, Poisson/Gamma log-link GLMs, logistic and softmax heads | Positive and Bayesian ridge, ARD regression, quantile, robust losses, SGD estimators, multinomial GLM, partial-fit, sparse inputs, and resident GPU kernels |
| Classification | Binary, softmax, OVR, OVO, multilabel, classifier chains, ordinal heads, five Naive Bayes variants, LDA/QDA, CART, random forest, Extra-Trees, linear/RBF SVM, calibrated heads, weighted Laplace or variational GP classifiers, coupled categorical variational GP classification, and the latent-Gaussian ordinal GP baseline | Sparse and multioutput labels, kernel SVM/SVR parity, calibrated cross-validation, additional coupled GP likelihoods, cloning, partial-fit, class-weight parity, and resident GPU training |
| Neighbors and clustering | Dense kNN and scalar/multi-output radius neighbors, seeded k-means, PCA, and a PCA-initialized linear autoencoder | KD-tree and ball-tree search, sparse and approximate search, density estimation, mixtures, incremental/randomized/kernel PCA, ICA, NMF, manifold methods, and outlier detectors |
| Trees and ensembles | CART, random forest, Extra-Trees, seeded CART bagging, binary and multiclass SAMME AdaBoost over weighted CART, stump boosting, exact and histogram XGBoost-style squared/logistic/Poisson/Tweedie/squared-log/Huber/quantile/absolute objectives, LightGBM-style leaf-wise growth, ranking, monotonic constraints, interaction groups, bounded ordered-gradient integer categorical partitions, sampling, versioned text/binary persistence, transactional fitted-prefix slicing, warm-start/validation contracts, and staged/contribution diagnostics | OOB/permutation/SHAP workflows, SAMME.R probability updates, categorical policies beyond ordered partitions, DART, GOSS, EFB, distributed workers, full warm-start LightGBM feature parity, and resident GPU histograms |
| Basis and pipelines | Polynomial, interaction-polynomial, Fourier, fixed-state random Fourier, radial, spline, callback, scaling, imputation, one-hot, horizontal/sequential/column pipelines, basis linear regression, joint basis-pipeline value/gradient/JVP/VJP/HVP products, and typed CPU/CUDA column-pipeline dispatch | Named DAGs, fan-out/fan-in and residual branches, sparse views, leakage-safe fit/transform, feature metadata routing, cloning, graph serialization, cross-validation, and resident transforms |

### Classification coverage audit

The shipped classification families are deliberately separate contracts rather
than one permissive catch-all. Each CPU row has an independent behavioral
oracle; unsupported accelerator or nonsmooth products return a typed status.

| Family | Implemented contract | Explicit remaining boundary |
| --- | --- | --- |
| Linear heads | Logistic, softmax, OVR, OVO, multilabel, classifier-chain, and cumulative-logit ordinal heads with sorted integer labels, weights where supported, probability/decision products, and deterministic ties | Sparse/partial-fit streams, full class-weight metadata routing, and resident GPU training |
| Probabilistic classifiers | Gaussian, Bernoulli, Multinomial, Complement, and Categorical Naive Bayes; weighted LDA/QDA; Platt, temperature, isotonic, reliability, and calibration metrics | Sparse state dictionaries, calibrated cross-validation, and broader probabilistic calibration policies |
| Margins and neighbors | Weighted linear and dense-RBF SVMs, one-class RBF SVM, exact dense kNN, and radius-neighbor classification | Kernel SVM/SVR parity, KD/ball-tree backends, sparse/approximate neighbors, and resident kernels |
| Trees and ensembles | Weighted CART, seeded random forest/Extra-Trees/bagging, SAMME AdaBoost, exact/histogram XGBoost-style binary/OVR/multiclass policies, and LightGBM leaf-wise binary/regression policies | SAMME.R, DART/GOSS/EFB, SHAP/OOB workflows, distributed training, and resident GPU histograms |
| GP classifiers | Binary/shared-kernel and OVR Laplace, weighted Bernoulli variational logistic/probit, coupled categorical variational, and latent-Gaussian ordinal GP baselines with packed/input products | Native cumulative ordinal likelihoods, likelihood/hyperparameter products through optimized modes, natural gradients, sparse/multitask classification, and resident GPU inference |
| Neural heads | MLP binary, multiclass, multilabel, ordinal, calibrated, Poisson, and physics-aware objectives with selected FortOpt L-BFGS-B and trajectory hypergradients | Convolutional/recurrent/attention/graph heads, focal/contrastive/sequence losses, distributed/mixed-precision training, and complete device derivatives |

The LightGBM contract now includes staged margins and linked predictions,
base-plus-tree additive contributions, and transactional fitted-prefix slicing.
The release benchmark records zero reconstruction error against an independent
tree-walk oracle. DART, GOSS, EFB, distributed workers, persistence, and
resident histogram execution remain open.

## Gaussian processes

| Surface | Current FortML surface | Required closure |
| --- | --- | --- |
| Kernels | RBF, Matérn, periodic, rational-quadratic, cosine, polynomial, linear, constant, white-noise, spectral-mixture, user, sum, and product kernels | Locally periodic, change-point, neural-network, graph, string, operator-valued, and physics-consistent kernels with compositional metadata |
| Observations | Exact value and first-derivative observations with prediction products, mixed covariance blocks, and selected second products | Higher derivative orders, registered linear operators, vector fields, Hessian observations, third-order query products, and operator-valued outputs |
| Inference | Dense exact, derivative, sparse, local, SKI, structured operators, binary/OVR and robust Poisson/Student-t Laplace paths, Bernoulli and coupled categorical variational classification, latent-Gaussian ordinal and dense Student-t/heteroskedastic process references, fixed-state variational kernel-log products, and batched multi-output posterior mean/input products with explicit CUDA refusal | Batch and multitask parameter products, LOVE/CIQ variance, whitened and unwhitened SVGP, natural gradients, stochastic minibatches, interdomain features, deep GPs, fantasy/online updates, and distributed inducing points |
| Likelihoods and state | Gaussian, Bernoulli/probit with weighted variational ELBOs, coupled variance-corrected categorical softmax ELBOs, selected Poisson/count objectives, latent-Gaussian ordinal probabilities, dense Student-t process regression with data-dependent covariance scaling, known-noise heteroskedastic GP regression, packed mean/noise/kernel state, and typed state/device boundaries | Native cumulative ordinal likelihoods and optimized cut points, multinomial, Student-t/heteroskedastic derivative and variational products, censored, warped, likelihood hyperparameter products, priors, constraints, state dictionaries, and resident GPU solves |

Every smooth product must expose value, gradient, JVP, VJP, and HVP where
mathematically defined. Products through a fitted mode or split policy must
state whether the state is fixed. Nonsmooth routing, active sets, and failed
factorizations return a named boundary status.

## Neural networks and training

| Surface | Current FortML surface | Required closure |
| --- | --- | --- |
| Modules | Dense MLPs, named sequential chains, BNN, VAE, vanilla RNN, Hamiltonian MLP, and a four-slot physics residual objective | Convolution, normalization, dropout, embeddings, attention, transformers, LSTM/GRU, temporal convolution, graph message passing, neural operators, HNN/LNN/SympNet, and physics-consistent modules |
| Losses | MSE, weighted binary and multiclass cross-entropy, Poisson log-rate, Huber/quantile products, selected physics residuals, and exact mixed HVPs | Focal, multilabel and count variants, contrastive, triplet, CTC, probabilistic reconstruction, full PDE residual catalog, and higher-order physics derivatives |
| Optimizers | SGD momentum/Nesterov, Adam/AdamW, Adagrad, RMSprop, unfactored Adafactor, AMSGrad and RAdam with validated moment checkpoints, Lion trajectory products, bounded FortOpt L-BFGS-B, schedules, gradient clipping, EMA, checkpoints, contiguous CPU optimizer groups, and fixed-SGD group-multiplier hypergradients | Matrix-factored Adafactor, complete AMSGrad trajectory products, cosine/plateau schedules, AMP with loss scaling, distributed state, migration, and resident multi-layer GPU training |
| Derivatives | Model parameter/input products and fixed-trajectory hypergradients for selected optimizers and schedules, with FortOpt consuming the same callbacks | A generated capability matrix for every parameter, input, loss, basis, kernel, likelihood, optimizer, schedule, validation decision, RNG state, and transfer counter, plus implicit derivatives through solves and optima |
| Initialization | Linear/PCA-seeded MLP state and a PCA-tied linear autoencoder | Polynomial, spline, Fourier, radial, GP-posterior, NNGP, NTK, Xavier/He, physics-consistent, symplectic, and Hamiltonian initialization with convergence and invariant benchmarks |

### Derivative and hyperparameter contract

“Differentiable” is recorded per state boundary; a model with discrete routing
does not pretend to differentiate its split decisions. The current contract is:

| State or operation | Current derivative evidence | Still required for full parity |
| --- | --- | --- |
| Smooth basis/pipeline maps | Analytic value, input JVP/VJP/HVP, packed parameter products, and FortOpt objective products on CPU | DAG routing, sparse/device products, and implicit fit derivatives |
| Dense MLP objectives | Parameter/input products, mixed HVPs, optimizer/schedule trajectory hypergradients, clipping/EMA/checkpoint state, and bounded FortOpt L-BFGS-B | Every module/loss/hyperparameter, stochastic/RNG derivatives, AMP/distributed products, and resident multi-layer GPU graphs |
| GP kernels and inference | Analytic kernel products, exact/derivative/mixed-observation products, selected variational/Laplace hyperproducts, and explicit finite-difference/adjoint oracles | Every derivative-observation order, likelihood/mode/inducing hyperparameter, multitask/operator-valued product, and resident solves |
| Trees and neighbors | Fixed-state prediction/contribution products where meaningful; split/routing, active-set, and neighbor membership changes return typed refusals | Smooth surrogate policies, full SHAP/OOB derivative contracts, and distributed/device implementations |
| Search and validation | Bounded FortOpt L-BFGS-B, generic search state, and selected objective hypergradients | Differentiable/Bayesian/multi-fidelity search parity, nested validation derivatives, and serialized trial graphs |

## Device and benchmark acceptance

Each claimed device path has three rows: a CPU reference, a resident device
execution, and a typed refusal when the backend is unavailable. The benchmark
records source revisions, compiler and driver versions, precision, residency,
transfer bytes, warmup, compile time, median and dispersion, memory, and an
independent oracle error. The external comparison matrix covers scikit-learn,
PyTorch, JAX, Flux, Lux, GPyTorch, GPflow, XGBoost, and LightGBM on matched
fixtures. Physics lanes add residual norm, conservation error, symplectic-form
defect, and short- and long-horizon trajectory error.

The current release evidence lives in `../fortml-bench`. Open rows remain
ROADMAP work packages until all acceptance fields are present.
