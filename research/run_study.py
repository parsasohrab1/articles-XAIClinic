"""Wisconsin Diagnostic Breast Cancer (n=569, 30 features): stability of explanations.

usage: python research/run_study.py /path/to/wdbc.data
For 100 bootstrap resamples, fit each model, compute three importance scores, then measure how
much the top-k feature set and the full ranking change between resamples.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from xaiclinic.stability import mean_jaccard, mean_kendall, selection_frequency, topk_sets

warnings.filterwarnings("ignore")
B, K = 100, 5


def main(path: str):
    df = pd.read_csv(path, header=None)
    y = (df[1] == "M").astype(int).to_numpy()
    X = df.iloc[:, 2:].to_numpy(float)
    names = [
        f"{b}_{s}"
        for s in ("mean", "se", "worst")
        for b in (
            "radius",
            "texture",
            "perimeter",
            "area",
            "smoothness",
            "compactness",
            "concavity",
            "concave_points",
            "symmetry",
            "fractal_dimension",
        )
    ]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    rng = np.random.default_rng(0)
    imps = {"logistic_abs_coef": [], "logistic_perm": [], "forest_gini": [], "forest_perm": []}
    accs = {"logistic": [], "forest": []}
    for b in range(B):
        idx = rng.integers(0, len(ytr), len(ytr))
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(
            Xtr[idx], ytr[idx]
        )
        rf = RandomForestClassifier(150, random_state=b, n_jobs=1).fit(Xtr[idx], ytr[idx])
        accs["logistic"].append(lr.score(Xte, yte))
        accs["forest"].append(rf.score(Xte, yte))
        imps["logistic_abs_coef"].append(np.abs(lr[-1].coef_[0]))
        imps["forest_gini"].append(rf.feature_importances_)
        imps["logistic_perm"].append(
            permutation_importance(lr, Xte, yte, n_repeats=3, random_state=b).importances_mean
        )
        imps["forest_perm"].append(
            permutation_importance(rf, Xte, yte, n_repeats=3, random_state=b).importances_mean
        )
    out = {
        "note": "real WDBC data; 70/30 split, 100 bootstrap refits of the training part; "
        f"top-{K} sets of 30 features",
        "n": len(y),
        "bootstrap_accuracy_mean": {k: float(np.mean(v)) for k, v in accs.items()},
        "methods": {},
    }
    for m, v in imps.items():
        A = np.array(v)
        sets = topk_sets(A, K)
        freq = selection_frequency(sets, A.shape[1])
        order = np.argsort(-freq)[:8]
        out["methods"][m] = {
            "topk_mean_jaccard": mean_jaccard(sets),
            "kendall_tau_mean": mean_kendall(A),
            "most_frequent_topk": [{"feature": names[j], "freq": float(freq[j])} for j in order],
        }
        print(
            m,
            round(out["methods"][m]["topk_mean_jaccard"], 3),
            round(out["methods"][m]["kendall_tau_mean"], 3),
            [(names[j], round(freq[j], 2)) for j in order[:5]],
        )
    # correlation structure that explains instability
    C = np.abs(np.corrcoef(X, rowvar=False))
    out["share_feature_pairs_abs_corr_gt_0.9"] = float((C[np.triu_indices(30, 1)] > 0.9).mean())
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
