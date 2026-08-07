# Validation and split iterators

`fortml_validation` provides deterministic index-only splitters for estimator
and pipeline workflows. They return one-based sample indices and never inspect
feature values. Fit each transformer on the returned training indices before
transforming the validation indices. This makes leakage visible at the call
site rather than hidden inside a model.

`kfold_splitter_t%initialize(n_samples,n_splits,status[,shuffle,seed])` creates
balanced folds. Call `next_split(train_indices,test_indices,has_split,status)`
until `has_split` is false, then call `reset()` to replay the same
sequence. Test folds differ in size by at most one sample. Optional shuffling
uses a local seeded Fisher--Yates stream.

`stratified_kfold_splitter_t%initialize(labels,n_splits,status[,shuffle,seed])`
assigns each class round-robin across folds, preserving deterministic class
counts as closely as the sample multiplicities allow. Its `next_split`,
`reset`, `sample_count`, `fold_count`, and `shuffled` methods have the same
contract. Empty or impossible folds, nonpositive seeds for shuffled iterators,
and calls before initialization return status errors.

The splitters own no feature, target, or transformer state. They are therefore
safe to reuse with sequential basis pipelines and with future cross-validation
or hyperparameter-search drivers.
