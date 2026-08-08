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
| Trees and ensembles | CART, random forest, Extra-Trees, seeded CART bagging, binary and multiclass SAMME AdaBoost over weighted CART, stump boosting, exact and histogram XGBoost-style squared/logistic/Poisson/Tweedie/squared-log/Huber/quantile/absolute objectives, LightGBM-style leaf-wise growth, ranking, monotonic constraints, interaction groups, bounded ordered-gradient integer categorical partitions, sampling, serialization, slicing, warm starts, and validation stopping | OOB/permutation/SHAP workflows, SAMME.R probability updates, categorical policies beyond ordered partitions, DART, GOSS, EFB, distributed workers, warm-start LightGBM, and resident GPU histograms |
| Basis and pipelines | Polynomial, interaction-polynomial, Fourier, fixed-state random Fourier, radial, spline, callback, scaling, imputation, one-hot, horizontal/sequential/column pipelines, basis linear regression, joint basis-pipeline value/gradient/JVP/VJP/HVP products, and typed CPU/CUDA column-pipeline dispatch | Named DAGs, fan-out/fan-in and residual branches, sparse views, leakage-safe fit/transform, feature metadata routing, cloning, graph serialization, cross-validation, and resident transforms |

The LightGBM contract now includes staged margins and linked predictions,
base-plus-tree additive contributions, and transactional fitted-prefix slicing.
The release benchmark records zero reconstruction error against an independent
tree-walk oracle. DART, GOSS, EFB, distributed workers, persistence, and
resident histogram execution remain open.

## Gaussian processes

| Surface | Current FortML surface | Required closure |
| --- | --- | --- |
| Kernels | RBF, Matérn, periodic, rational-quadratic, cosine, polynomial, linear, constant, white-noise, spectral-mixture, user, sum, and product kernels | Locally periodic, change-point, neural-network, graph, string, operator-valued, and physics-consistent kernels with compositional metadata |
| Observations | Exact value and first-derivative observations with prediction products, mixed covariance blocks, selected second products, and the scalar 1-D RBF `second_derivative_gp_t` value/gradient/Hessian observation reference | Higher derivative orders beyond two, registered linear operators, vector fields, operator-valued outputs, and resident derivative covariance/solve kernels |
| Inference | Dense exact, derivative, sparse, local, SKI, structured operators, binary/OVR Laplace, Bernoulli and coupled categorical variational classification, latent-Gaussian ordinal and dense Student-t/heteroskedastic process references, fixed-state variational kernel-log products, and batched multi-output posterior mean/input products with explicit CUDA refusal | Batch and multitask parameter products, LOVE/CIQ variance, whitened and unwhitened SVGP, natural gradients, stochastic minibatches, interdomain features, deep GPs, fantasy/online updates, and distributed inducing points |
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
| Optimizers | SGD momentum/Nesterov, Adam/AdamW, Adagrad, RMSprop, unfactored Adafactor, AMSGrad and RAdam with validated moment checkpoints, Lion trajectory products, bounded FortOpt L-BFGS-B, schedules, checkpoints, contiguous CPU optimizer groups, and fixed-SGD group-multiplier hypergradients | Matrix-factored Adafactor, AMSGrad trajectory products, cosine/plateau schedules, AMP with loss scaling, gradient accumulation, EMA, distributed state, migration, and resident multi-layer GPU training |
| Derivatives | Model parameter/input products and fixed-trajectory hypergradients for selected optimizers and schedules, with FortOpt consuming the same callbacks | A generated capability matrix for every parameter, input, loss, basis, kernel, likelihood, optimizer, schedule, validation decision, RNG state, and transfer counter, plus implicit derivatives through solves and optima |
| Initialization | Linear/PCA-seeded MLP state and a PCA-tied linear autoencoder | Polynomial, spline, Fourier, radial, GP-posterior, NNGP, NTK, Xavier/He, physics-consistent, symplectic, and Hamiltonian initialization with convergence and invariant benchmarks |

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
