"""Aggregate sentence labels to the call level.

Input is the labeled sentence file from 07_label_corpus.py. Output has one row per call:

    n_temporal   number of sentences with a temporal label (not s)
    diversity    mean over temporal sentences of 1 (one orientation), 2 (two), 3 (three)
"""
from pathlib import Path

import pandas as pd

import global_options


def aggregate(df, call_id):
    """Compute the call-level measures.

    Arguments:
        df {pd.DataFrame} -- labeled sentences (column label)
        call_id {[str]} -- columns identifying a call

    Returns:
        pd.DataFrame -- one row per call
    """
    d = df[df["label"] != "s"].copy()
    d["n_orient"] = d["label"].astype(str).str.split("+").str.len()
    g = d.groupby(call_id)
    out = pd.DataFrame({"n_temporal": g.size(), "diversity": g["n_orient"].mean()})
    return out.reset_index()


if __name__ == "__main__":
    in_file = Path(global_options.OUTPUT_FOLDER, "ceo_sentences_labeled.csv")
    df = pd.read_csv(in_file)
    print("Loaded {} labeled sentences".format(len(df)))
    calls = aggregate(df, global_options.CALL_ID)
    print("{} calls".format(len(calls)))
    out_file = Path(global_options.OUTPUT_FOLDER, "call_measures.csv")
    calls.to_csv(out_file, index=False)
    print(calls.describe().T[["count", "mean", "std", "min", "max"]])
    print("Saved to", out_file)
