# Multiclass SAMME.R AdaBoost

`adaboost_classifier_t` supports the deterministic probability-update variant
through `algorithm=ADABOOST_ALGORITHM_SAMME_R`. It requires at least three
classes and weighted CART weak learners. Each stage clips the tree's aligned
class probabilities at `1e-12`, centres their logarithms, and updates each
sample weight by

```
exp(mean(log(p)) - log(p_true))
```

After fitting, `decision_function` accumulates `(K-1)` times the centred
log-probabilities. `predict_proba` applies the matching `1/(K-1)` softmax
scale, giving the normalized geometric ensemble of the weak-learner
probabilities. Stage weights are exposed as unit values; `algorithm()` returns
`ADABOOST_ALGORITHM_SAMME_R`.

The fitted state is committed only after every stage and normalization succeeds.
Malformed refits preserve the previous model. CART split routing is
nondifferentiable, so input JVPs and parameter products return typed
`FORTNUM_NOT_IMPLEMENTED` statuses. CPU prediction is supported; CUDA requests
return an output-preserving typed refusal until a resident tree kernel exists.
The independent source oracle is `test_adaboost_samme_r`; the release benchmark
is `fortml-bench/scripts/bench_adaboost_samme_r.py`.
