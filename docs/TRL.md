# TRL assessment – XAIClinic

**Claimed level: TRL 4** (laboratory validation on one public real dataset). Not TRL 5: no clinical-environment validation, no user study.

Data: Wisconsin Diagnostic Breast Cancer, n = 569; held-out accuracy over bootstrap refits: logistic 0.954, forest 0.941. Top-5 of 30 features; 100 bootstrap refits.

| Importance | Mean top-5 Jaccard | Mean Kendall tau | Most frequent top-5 features (frequency) |
|---|---|---|---|
| logistic_abs_coef | 0.325 | 0.496 | radius_se (0.83), radius_worst (0.73), symmetry_worst (0.51), smoothness_worst (0.48), perimeter_se (0.44) |
| logistic_perm | 0.269 | 0.363 | texture_worst (0.62), radius_worst (0.61), radius_se (0.58), texture_mean (0.49), compactness_se (0.47) |
| forest_gini | 0.792 | 0.749 | perimeter_worst (0.97), concave_points_mean (0.96), concave_points_worst (0.92), radius_worst (0.90), area_worst (0.89) |
| forest_perm | 0.207 | 0.135 | area_worst (0.66), radius_worst (0.62), perimeter_worst (0.44), texture_worst (0.42), area_se (0.27) |

## Findings and caveats
- Both classifiers are accurate (about 0.94–0.95) yet their explanations differ in stability: only random-forest Gini importance gives a reproducible top-5 (Jaccard 0.79); logistic coefficients (0.33) and both permutation-importance variants (0.27 and 0.21) are not.
- The 30 features are strongly collinear (radius, perimeter, area at three levels); the model can substitute one for another between resamples, so *which* feature is named changes while the predictions do not. A different top-5 list is therefore not evidence that the model changed.
- Permutation importance here is computed on a fixed test set of 171 patients with 3 repeats, so part of its instability is estimator noise rather than model instability; the two sources are not separated in this study.
- Gini importance is known to be biased toward high-variability features; its stability does not mean it is correct.
- One dataset, one split seed; the conclusion is about this dataset only. No claim is made that the named features are clinically meaningful.
