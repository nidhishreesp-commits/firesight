"""Train LR and MLP on verifier features; 5-fold GroupKFold by source image; pick by AUC."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="build/verifier/features_train.npz")
    ap.add_argument("--out", default="build/verifier/model.joblib")
    args = ap.parse_args()
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.pipeline import make_pipeline
    import joblib

    d = np.load(Path(args.features), allow_pickle=True)
    X, y, groups = d["X"], d["y"], d["groups"]
    print("X", X.shape, "pos", int(y.sum()), "neg", int((1-y).sum()), "groups", len(set(groups)))
    gkf = GroupKFold(n_splits=5)
    models = {
        "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
        "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 32), max_iter=300,
                                                                    early_stopping=True, random_state=0)),
    }
    scores = {}
    for name, factory in models.items():
        aucs, aps = [], []
        for tr, te in gkf.split(X, y, groups):
            m = factory(); m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p)); aps.append(average_precision_score(y[te], p))
        scores[name] = {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                        "ap_mean": float(np.mean(aps)), "folds": aucs}
        print(name, json.dumps(scores[name]))
    best = max(scores, key=lambda k: scores[k]["auc_mean"])
    print("best", best)
    model = models[best](); model.fit(X, y)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "kind": best, "cv": scores}, Path(args.out))
    print("saved", args.out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
