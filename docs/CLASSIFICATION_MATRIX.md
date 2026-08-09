# Classification acceptance matrix

This matrix records the classification families that FortML exposes and the
evidence required before a family is counted as production parity. A CPU
implementation with a typed CUDA refusal is a valid bounded slice. It is not a
claim of accelerator parity.

| Family | FortML surface | Current products | Open parity gate |
| --- | --- | --- | --- |
| Binary linear | `logistic_regression_t`, `linear_svm_classifier_t` | Weighted fit, sorted labels, decision/probability values, packed parameter JVP/VJP, selected HVP and FortOpt objectives | SGD solver variants, kernel SVM, sparse/device training, complete HVP matrix |
| Multinomial linear | `softmax_regression_t`, `softmax_training_objective_t` | Stable weighted softmax, sorted classes, parameter/input products, L-BFGS-B and mixed L2 products | OVR/OVO coupling products through every optimizer and sparse/device paths |
| OVR and OVO | `ovr_logistic_classifier_t`, `ovo_logistic_classifier_t` | Deterministic class/pair ordering, weighted fits, probability policy, packed products | Pairwise coupling, calibration during fit, sparse and resident GPU paths |
| Naive Bayes | Gaussian, Bernoulli, Multinomial, Complement, Categorical modules | Weighted moments/counts, smoothing, stable log-probabilities, input/parameter products and refusal tests | Incremental and sparse count workflows, full multiclass HVPs |
| Multilabel and chains | `multilabel_logistic_classifier_t`, `classifier_chain_logistic_classifier_t`, MLP/GP multilabel wrappers | Indicator validation, per-label thresholds, chain metadata, weighted products and typed device boundaries | Sparse indicators, regressor chains, calibrated multilabel uncertainty |
| Ordinal | `ordinal_logistic_classifier_t`, `mlp_ordinal_classifier_t`, `gp_ordinal_classification_t` | Ordered labels, cumulative logits, threshold metadata, input/parameter products and kink refusals | Native ordinal GP likelihood, optimized cut points, complete HVPs |
| Trees and boosting | CART, random/extra forests, bagging, AdaBoost, XGBoost and LightGBM adapters | Weighted fits, missing routing, constraints, staged margins, persistence, binary and multiclass stable log-probabilities for XGBoost and LightGBM, packed input and leaf-coordinate products, multi-output adapters, typed CUDA boundaries | SAMME.R, EFB, distributed growth, SHAP interactions, resident GPU histograms |
| Calibration | Logistic, softmax, temperature, Platt, isotonic, calibrated MLP | OOF routing, weighted policies, transactional refits, fixed-state products where smooth | Calibrated GP and multilabel workflows, resident calibration kernels |
| Gaussian-process classifiers | Binary/shared-kernel, OVR, and independent multilabel Laplace wrappers, variational Bernoulli, coupled categorical, and ordinal baselines. Stable binary/OVR/multilabel log probabilities have fixed-state input and packed-kernel JVP/VJP products. Independent multilabel kernel blocks also have a shared FortOpt L-BFGS-B objective | Sorted labels, predictive probabilities and log probabilities, inducing-state products, likelihood temperature products, shared and independent FortOpt slices, typed CPU/CUDA dispatch, and independent log-probability normalization oracles | Full likelihood catalog, natural gradients, batch/multitask state, complete likelihood/kernel HVPs, resident inference |
| Neural classifiers | MLP binary, multiclass, multilabel, ordinal and calibrated adapters | Deterministic Adam/minibatches, weighting, named parameters, probability products, exact loss HVPs and FortOpt objectives, transactional FP64 loss-scale products with overflow backoff | Full module tree, master-weight FP16/BF16 execution, distributed state, convolution/attention/sequence/graph heads, resident multilayer training |

Every row requires the same evidence bundle: an independent value oracle, an
adjoint or finite-difference derivative check for each claimed product,
transactional malformed-input tests, a device capability row, and a raw
benchmark record with source and benchmark revisions. The detailed long-term
register is in [`ROADMAP.md`](../ROADMAP.md).
