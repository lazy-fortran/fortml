# Classifier-chain probability HVP

`classifier_chain_t%predict_proba_hvp` differentiates the fixed fitted
probability graph in a joint direction of packed head parameters and inputs.
For a probability cotangent `u`, direction `(theta_dot, x_dot)`, and outputs
`(theta_hvp, x_hvp)`, it computes the forward-over-reverse product

```text
d [ VJP_(theta,x) sum(u * predict_proba(x; theta)) ]
```

The reverse pass carries the cotangent through the previous positive
probabilities used as smooth chain features. Each logistic head uses the exact
sigmoid first and second derivatives, including the bilinear parameter/input
term in its score. The packed output order is the same as `parameters()`:
head coefficient blocks followed by optional intercepts, in output order.

The operation is fixed-state. It does not refit a head, differentiate class
sorting, or differentiate thresholded labels. `predict_proba_hvp_device` runs
the exact CPU path for a selected CPU device and returns
`FORTNUM_NOT_IMPLEMENTED` for CUDA until a resident multi-head kernel exists;
it never hides a host fallback or transfer.

`test_classifier_chain` checks the product against a central finite difference
of the existing parameter and input VJPs using an independent cotangent and
joint direction. The release benchmark records separate parameter and input
HVP errors in `fortml-bench/results/classifier_chain.csv`.
