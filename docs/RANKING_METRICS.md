# Grouped ranking metrics

fortml_ranking_metrics provides a standalone grouped NDCG reduction for tree
validation and downstream ranking workflows. ranking_ndcg accepts arbitrary
positive integer query IDs, real relevance values, raw margins, an optional
per-query cutoff k, nonnegative sample weights, and a gain base greater than
one. Query IDs need not be contiguous. Score ties retain input order, which
makes the metric reproducible across repeated runs.

The ideal list is computed independently for every query by descending
weighted gain. The returned value is the macro mean over queries with positive
ideal gain. A query with zero ideal gain is excluded, while an all-zero
fixture is refused so callers cannot mistake an undefined metric for zero.
The CPU implementation does not depend on a fitted XGBoost or LightGBM model.

ranking_ndcg_device dispatches CPU calls to the same reduction. A selected CUDA
context returns FORTNUM_NOT_IMPLEMENTED until a resident grouped reduction is
linked. The API does not use a hidden host fallback.

The independent behavioral oracle in test/test_ranking_metrics.f90 checks a
hand-computed two-query NDCG, per-query cutoff, weighted bounded output,
zero-ideal validation, CPU dispatch, and typed CUDA refusal. The release
workload is fortml_bench_ranking_metrics.
