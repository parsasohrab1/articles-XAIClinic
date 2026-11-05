# Software Requirements Specification – XAIClinic

- SR-1 `stability.topk_sets` returns the top-k feature set for every resample; `mean_jaccard` gives the mean pairwise Jaccard index of those sets.
- SR-2 `stability.mean_kendall` gives the mean pairwise Kendall tau of the full importance rankings.
- SR-3 `stability.selection_frequency` gives how often each feature enters the top-k.
- SR-4 `research/run_study.py`: 70/30 stratified split (seed 0); 100 bootstrap refits of the training part; logistic regression (standardised) and a 150-tree random forest; importances computed on every refit; permutation importance uses the fixed held-out part (3 repeats).
- SR-5 Three unit tests.
