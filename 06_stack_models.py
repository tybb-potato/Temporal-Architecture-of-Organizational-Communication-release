"""Combine the trained members with a logistic regression stacker.

Each member contributes a block of features on the held-out gold test set: the
encoders their four component probabilities, the LLMs their label split into the same
four components (0/1). A multinomial logistic regression maps the concatenated features
to the eight labels. Accuracy is estimated out of fold on the held-out set, with one
stratified k-fold split for each of STACK_REPEATS seeds, so the stacker is never scored on
rows it was fit on; the sd is taken over the seeds. The final stacker is fit on all rows
and saved for 07_label_corpus.py.

    python 06_stack_models.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

import global_options


def member_features():
    """Load every member's held-out output; all members score gold_test.csv in the same order.

    Returns:
        np.ndarray, np.ndarray, [str] -- feature matrix, gold labels, feature names
    """
    blocks, names, base = [], [], None
    for member in global_options.ENCODERS:
        d = pd.read_csv(Path(global_options.MODEL_FOLDER, member, "test_probs.csv"))
        base = d if base is None else base
        assert d[global_options.SENT_COL].equals(base[global_options.SENT_COL]), member + " was scored on a different test set"
        blocks.append(d[["p_" + c for c in global_options.COMPONENTS]].values)
        names += ["{}_{}".format(member, c) for c in global_options.COMPONENTS]
        print("{:<18} held-out accuracy {:.4f}".format(member, (d["pred"] == d["gold_label"]).mean()))
    for member in global_options.LLMS:
        d = pd.read_csv(Path(global_options.MODEL_FOLDER, member, "test_preds.csv"), keep_default_na=False)
        assert d[global_options.SENT_COL].equals(base[global_options.SENT_COL]), member + " was scored on a different test set"
        blocks.append(np.array([[float(c in p.split("+")) for c in global_options.COMPONENTS] for p in d["pred"]]))
        names += ["{}_{}".format(member, c) for c in global_options.COMPONENTS]
        print("{:<18} held-out accuracy {:.4f}".format(member, (d["pred"] == d["gold_label"]).mean()))
    return np.hstack(blocks), base["gold_label"].values, names


if __name__ == "__main__":
    X, y, names = member_features()
    print("{} rows, {} features".format(*X.shape))

    # stratified folds need at least STACK_FOLDS rows per label; fall back to plain folds otherwise
    folds = StratifiedKFold if pd.Series(y).value_counts().min() >= global_options.STACK_FOLDS else KFold
    accs = []  # one out-of-fold accuracy per seed: every row is predicted once by a stacker not fit on it
    for seed in range(global_options.STACK_REPEATS):
        pred = np.empty(len(y), dtype=object)
        for tr, te in folds(n_splits=global_options.STACK_FOLDS, shuffle=True, random_state=seed).split(X, y):
            pred[te] = LogisticRegression(C=global_options.STACK_C, max_iter=2000).fit(X[tr], y[tr]).predict(X[te])
        accs.append((pred == y).mean())
    print("Stacker out-of-fold accuracy: {:.4f} (sd over {} seeds {:.4f})".format(np.mean(accs), len(accs), np.std(accs)))

    clf = LogisticRegression(C=global_options.STACK_C, max_iter=2000).fit(X, y)
    out_file = Path(global_options.MODEL_FOLDER, "stacker.joblib")
    joblib.dump({"model": clf, "features": names}, out_file)
    json.dump({"oof_accuracy": float(np.mean(accs)), "oof_sd": float(np.std(accs))},
              open(Path(global_options.MODEL_FOLDER, "stacker_metrics.json"), "w"), indent=2)
    print("Saved to", out_file)
