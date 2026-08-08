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
| Classification | Binary, softmax, OVR, OVO, multilabel, classifier chains, ordinal heads, five Naive Bayes variants, LDA/QDA, CART, random forest, Extra-Trees, linear/RBF SVM, calibrated heads, leakage-safe binary and multiclass softmax calibration from stratified out-of-fold margins, weighted Laplace or variational GP classifiers, coupled categorical variational GP classification, and the latent-Gaussian ordinal GP baseline | Sparse and multioutput labels, kernel SVM/SVR parity, additional coupled GP likelihoods, cloning, partial-fit, class-weight parity, and resident GPU training |
| Neighbors and clustering | Dense kNN and scalar/multi-output radius neighbors, seeded k-means, PCA, and a PCA-initialized linear autoencoder | KD-tree and ball-tree search, sparse and approximate search, density estimation, mixtures, incremental/randomized/kernel PCA, ICA, NMF, manifold methods, and outlier detectors |
| Trees and ensembles | CART, random forest with stored bootstrap inclusion, transactional OOB decision probabilities/accuracy/coverage and explicit insufficient-coverage refusal, deterministic fixed-state accuracy permutation importance with a NumPy replay oracle, Extra-Trees, seeded CART bagging, binary and multiclass SAMME plus multiclass SAMME.R probability-update AdaBoost over weighted CART, stump boosting, exact and histogram XGBoost-style squared/logistic/Poisson/fixed-shape Gamma/Tweedie/squared-log/Huber/quantile/absolute objectives, validation-aware multiclass OVR training with weighted log-loss and best-prefix metadata, LightGBM-style leaf-wise growth with deterministic GOSS top/other-rate gradient/Hessian reweighting and bounded seeded DART/dropout tree normalisation, ranking, monotonic constraints, interaction groups, bounded ordered-gradient integer categorical partitions, sampling, versioned text/binary persistence, transactional fitted-prefix slicing, warm-start/validation contracts, staged/contribution diagnostics, and bounded exact-subset SHAP-like per-feature raw-margin attributions for XGBoost and LightGBM | Monotone prediction checks, partial dependence/model export diagnostics, categorical policies beyond ordered partitions, EFB, distributed workers, full warm-start LightGBM feature parity, full SHAP interaction/explanation parity, resident GPU histograms, and differentiable routing |
| Basis and pipelines | Polynomial, interaction-polynomial, Chebyshev, Fourier, fixed-state random Fourier, radial, spline, callback, scaling, imputation, one-hot, transactional dense input schemas with unique feature names, horizontal/sequential/column pipelines, bounded named fan-out/fan-in and main-plus-residual-sum DAG branches, basis linear regression, joint basis-pipeline value/gradient/JVP/VJP/HVP products, versioned transactional persistence for preconfigured fitted pipelines, typed CPU/CUDA dispatch, and chronological expanding/rolling time-series splitters with scorer and clone/reset metadata | Conditional branches, sparse views, leakage-safe fit/transform, estimator-wide metadata routing, model-specific clone implementation, structural graph serialization, repeated/Monte Carlo cross-validation scoring, and resident transforms |

### Classification coverage audit

The shipped classification families are deliberately separate contracts rather
than one permissive catch-all. Each CPU row has an independent behavioral
oracle; unsupported accelerator or nonsmooth products return a typed status.

| Family | Implemented contract | Explicit remaining boundary |
| --- | --- | --- |
| Linear heads | Logistic, softmax, OVR, OVO, multilabel, classifier-chain, and cumulative-logit ordinal heads with sorted integer labels, weights where supported, probability/decision products, and deterministic ties | Sparse/partial-fit streams, full class-weight metadata routing, and resident GPU training |
| Probabilistic classifiers | Gaussian, Bernoulli, Multinomial, Complement, and Categorical Naive Bayes; weighted LDA/QDA; Platt, temperature, isotonic, reliability, and calibration metrics; binary calibration plus weighted one-vs-rest multiclass temperature/Platt/isotonic maps with sorted classes, simplex probabilities, smooth products for temperature/Platt, and fold/log-loss diagnostics where OOF wrappers apply | Sparse state dictionaries, OOF multiclass Platt routing, and broader probabilistic calibration policies |
| Margins and neighbors | Weighted linear and dense-RBF SVMs, one-class RBF SVM, exact dense kNN, and radius-neighbor classification | Kernel SVM/SVR parity, KD/ball-tree backends, sparse/approximate neighbors, and resident kernels |
| Trees and ensembles | Weighted CART, seeded random forest/Extra-Trees/bagging, stored random-forest bootstrap inclusion, transactional OOB probabilities/score/coverage, typed insufficient-coverage status, fixed-state deterministic accuracy permutation importance with transactional CPU/CUDA refusal paths, SAMME and SAMME.R AdaBoost, exact/histogram XGBoost-style binary/OVR/multiclass policies including validation-aware best-prefix metadata, LightGBM leaf-wise binary/regression policies with deterministic GOSS and bounded seeded DART/dropout tree normalisation, and bounded exact-subset SHAP-like raw-margin attributions for both boosted-tree families | SHAP interaction/deep-explanation parity, EFB, distributed training, differentiable routing, and resident GPU histograms |
| GP classifiers | Binary/shared-kernel and OVR Laplace, weighted Bernoulli variational logistic/probit, coupled categorical variational, and latent-Gaussian ordinal GP baselines with packed/input products; categorical likelihood temperature JVP/VJP/HVP products and binary implicit-mode kernel HVPs are independently gated | Native cumulative ordinal likelihoods, broader likelihood/hyperparameter products through optimized modes, natural gradients, sparse/multitask classification, and resident GPU inference |
| Neural heads | MLP binary, multiclass, multilabel, ordinal, calibrated, Poisson, and physics-aware objectives with selected FortOpt L-BFGS-B and trajectory hypergradients; multiclass probability heads now expose fixed-input packed-parameter JVP/VJP products with explicit CPU/CUDA dispatch; focal-softmax value/JVP/VJP/HVP products are wired through the multiclass MLP and FortOpt objectives | Convolutional/recurrent/attention/graph heads, contrastive/sequence losses, distributed/mixed-precision training, and complete resident device derivatives |
 | GP classifiers | Binary/shared-kernel and OVR Laplace, weighted Bernoulli variational logistic/probit, coupled categorical variational, and latent-Gaussian ordinal GP baselines with packed/input products; categorical likelihood temperature JVP/VJP/HVP products and binary implicit-mode kernel HVPs are independently gated | Native cumulative ordinal likelihoods, broader likelihood/hyperparameter products through optimized modes, natural gradients, sparse/multitask classification, and resident GPU inference |
 | Neural heads | MLP binary, multiclass, multilabel, ordinal, calibrated, Poisson, and physics-aware objectives with selected FortOpt L-BFGS-B and trajectory hypergradients; multiclass probability heads now expose fixed-input packed-parameter JVP/VJP products with explicit CPU/CUDA dispatch; focal-softmax and pairwise contrastive value/JVP/VJP/HVP products are wired through shared losses and MLP/FortOpt seams, with weighted reductions and typed kink/CUDA boundaries | Convolutional/recurrent/attention/graph heads, triplet/sequence losses, distributed/mixed-precision training, and complete resident device derivatives |

The LightGBM contract now includes staged margins and linked predictions,
base-plus-tree additive contributions, transactional fitted-prefix slicing,
schema-3 persistence, deterministic GOSS top/other-rate sampling, and bounded
seeded DART/dropout tree normalisation with an independent hand oracle. The
release benchmarks record zero reconstruction error against independent
tree-walk oracles. EFB, distributed workers, and resident histogram execution
remain open.

## Gaussian processes

| Surface | Current FortML surface | Required closure |
| --- | --- | --- |
| Kernels | RBF, Matérn, periodic, locally periodic, change-point, rational-quadratic, cosine, polynomial, linear, constant, white-noise, spectral-mixture, user, sum, and product kernels | Neural-network, graph, string, operator-valued, and physics-consistent kernels with compositional metadata |
| Observations | Exact value and first-derivative observations with prediction products, mixed covariance blocks, selected second products, and the scalar 1-D RBF/Matérn-5/2 `second_derivative_gp_t` value/gradient/Hessian reference; RBF additionally supports third-derivative observations, order-six covariance, order-seven query products, and analytic likelihood gradient/HVP | Higher derivative orders beyond RBF order three, registered linear operators, vector fields, operator-valued outputs, Matérn parameter jets, and resident derivative covariance/solve kernels |
| Inference | Dense exact, derivative, sparse, local, SKI, structured operators, binary/OVR and robust Poisson/Student-t Laplace paths, Bernoulli and coupled categorical variational classification, latent-Gaussian ordinal and dense Student-t/heteroskedastic process references, fixed-state variational kernel-log products, batched multi-output posterior mean/input products, and a CPU deep-kernel GP composition with exact feature/weight likelihood gradients and explicit CUDA refusal | Batch and multitask parameter products, LOVE/CIQ variance, whitened and unwhitened SVGP, natural gradients, stochastic minibatches, interdomain features, jointly trained deep GPs/KISS-GP, fantasy/online updates, and distributed inducing points |
| Likelihoods and state | Gaussian, Bernoulli/probit with weighted variational ELBOs, coupled variance-corrected categorical softmax ELBOs with positive temperature products and likelihood-only fitting, selected Poisson/count objectives, latent-Gaussian ordinal probabilities, dense Student-t process regression with data-dependent covariance scaling, known-noise heteroskedastic GP regression, packed mean/noise/kernel state, fixed-state sparse-GP Gaussian-likelihood log-noise ELBO JVP/VJP/HVP products with transactional transformed setters, and typed state/device boundaries | Native cumulative ordinal likelihoods and optimized cut points, multinomial, Student-t/heteroskedastic derivative and variational products, censored, warped, broader likelihood hyperparameter products, priors, constraints, state dictionaries, and resident GPU solves |

Every smooth product must expose value, gradient, JVP, VJP, and HVP where
mathematically defined. Products through a fitted mode or split policy must
state whether the state is fixed. Nonsmooth routing, active sets, and failed
factorizations return a named boundary status.

## Neural networks and training

| Surface | Current FortML surface | Required closure |
| --- | --- | --- |
| Modules | Dense MLPs, named sequential chains, BNN, VAE, vanilla RNN, Hamiltonian MLP, a four-slot physics residual objective, and reusable canonical symplectic-form residual/value/JVP/VJP diagnostics with a `physics_constraint_t` adapter | Convolution, normalization, dropout, embeddings, attention, transformers, LSTM/GRU, temporal convolution, graph message passing, neural operators, HNN/LNN/SympNet, and physics-consistent modules |
| Losses | MSE, weighted binary and multiclass cross-entropy, stable softmax/log-softmax value/JVP/VJP/HVP products, weighted softmax-cross-entropy products, focal binary and multiclass focal-softmax value/JVP/VJP/HVP products, Poisson log-rate, Huber/quantile products, selected physics residuals, canonical symplectic-form least-squares residuals with exact first-order products, and exact mixed HVPs | Multilabel focal variants, Poisson dispersion/count extensions, contrastive, triplet, CTC, probabilistic reconstruction, full PDE residual catalog, higher-order physics derivatives, and resident CUDA loss kernels |
| Optimizers | SGD momentum/Nesterov, Adam/AdamW, Adagrad, RMSprop, unfactored and layout-aware matrix-factored Adafactor, AMSGrad and RAdam with validated moment checkpoints, Lion in both the generic objective trainer and MLP trainer, Lion trajectory products, fixed full-batch AMSGrad trajectory products with max-state active-set refusals, bounded FortOpt L-BFGS-B, schedules, exact affine outer HVPs for constant/warmup/cosine/exponential/one-cycle schedule coordinates, gradient clipping, EMA, checkpoints, contiguous CPU optimizer groups, and fixed-SGD group-multiplier hypergradients | Schedule derivatives through optimizer groups and validation policies, AMP with loss scaling, distributed state, and resident multi-layer GPU training |
| Derivatives | Model parameter/input products and fixed-trajectory hypergradients for selected optimizers and schedules, with exact outer SGD momentum/Nesterov HVPs on one-layer affine MLPs and FortOpt consuming the same callbacks | A generated capability matrix for every parameter, input, loss, basis, kernel, likelihood, optimizer, schedule, validation decision, RNG state, and transfer counter, plus implicit derivatives through solves and optima |
| Initialization | Checked affine `initialize_linear` state, explicit `initialize_from_pca` centered/whitened linear MLP optimum, a PCA-tied linear autoencoder, and finite-feature GP/NTK last-layer kernel-ridge initialization | Polynomial, spline, Fourier, radial, full GP-posterior, NNGP, physics-consistent, symplectic, and Hamiltonian initialization with convergence and invariant benchmarks |

### Derivative and hyperparameter contract

“Differentiable” is recorded per state boundary; a model with discrete routing
does not pretend to differentiate its split decisions. The current contract is:

| State or operation | Current derivative evidence | Still required for full parity |
| --- | --- | --- |
| Smooth basis/pipeline maps | Analytic value, input JVP/VJP/HVP, packed parameter products, bounded named fan-out/fan-in routing, and FortOpt objective products on CPU | Residual/conditional DAG routing, sparse/device products, and implicit fit derivatives |
| Dense MLP objectives | Parameter/input products, mixed HVPs, optimizer/schedule trajectory hypergradients including exact one-cycle peak/final products, exact affine one-layer outer SGD HVPs, clipping/EMA/checkpoint state, and bounded FortOpt L-BFGS-B | Every module/loss/hyperparameter, optimizer-group/validation-policy/device-state schedule derivatives, stochastic/RNG derivatives, AMP/distributed products, and resident multi-layer GPU graphs |
| GP kernels and inference | Analytic kernel products, exact/derivative/mixed-observation products (including the spectral-mixture parameter HVP), selected variational/Laplace hyperproducts, binary Laplace implicit-mode kernel HVP, and explicit finite-difference/adjoint oracles | Every derivative-observation order, likelihood/mode/inducing hyperparameter, multitask/operator-valued product, and resident solves |
| Trees and neighbors | Fixed-state prediction/contribution products where meaningful; split/routing, active-set, and neighbor membership changes return typed refusals | Smooth surrogate policies, full SHAP/OOB derivative contracts, and distributed/device implementations |
| Search and validation | Bounded FortOpt L-BFGS-B, generic search state, deterministic K-fold/stratified/grouped/chronological splitters, scorer orientation metadata, and explicit clone/reset declarations with selected objective hypergradients | Repeated/Monte Carlo cross-validation scoring, differentiable/Bayesian/multi-fidelity search parity, nested validation derivatives, and serialized trial graphs |

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
