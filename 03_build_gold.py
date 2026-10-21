"""Build the human-audited gold set.

Before human coding, "sample" draws the sentences for the coders from the LLM-labeled
file and writes the audit sheet. After human coding, "split" divides the coded sentences
into a training set and a held-out test set, and "silver" writes the LLM-labeled
sentences that were not audited as extra training data for the encoders.

    python 03_build_gold.py sample
    python 03_build_gold.py split
    python 03_build_gold.py silver
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import global_options


def draw_audit_sample(df, n, seed):
    """Draw a random sample of the LLM-labeled sentences for human coding.

    Arguments:
        df {pd.DataFrame} -- LLM-labeled sentences (column llm_label)
        n {int} -- number of sentences to draw

    Returns:
        pd.DataFrame -- audit sheet with an empty gold_label column
    """
    df = df[df["llm_label"].isin(global_options.LABELS)]
    sample = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    sample["gold_label"] = ""
    sample["coder_note"] = ""
    return sample


def split_gold(df, test_size, seed):
    """Stratified train/test split of the audited sentences. Rows without a valid
    gold_label are dropped.
    """
    df = df[df["gold_label"].isin(global_options.LABELS)].copy()
    # stratification needs at least two sentences per label; fall back to a plain split otherwise
    strat = df["gold_label"] if df["gold_label"].value_counts().min() >= 2 else None
    train, test = train_test_split(df, test_size=test_size, stratify=strat, random_state=seed)
    return train, test


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    out = Path(global_options.OUTPUT_FOLDER)
    sheet_file = out / "audit_sheet.csv"
    if what == "sample":
        assert not sheet_file.exists(), "{} exists; move it away before drawing a new sample".format(sheet_file)
        df = pd.read_csv(out / "llm_labels.csv", dtype=str, keep_default_na=False)
        sheet = draw_audit_sample(df, global_options.N_AUDIT, global_options.SEED)
        sheet.to_csv(sheet_file, index=False)
        print("Audit sheet with {} sentences saved to {}".format(len(sheet), sheet_file))
        print(sheet["llm_label"].value_counts())
    elif what == "split":
        df = pd.read_csv(sheet_file, dtype=str, keep_default_na=False)
        train, test = split_gold(df, global_options.TEST_SIZE, global_options.SEED)
        train.to_csv(out / "gold" / "gold_train.csv", index=False)
        test.to_csv(out / "gold" / "gold_test.csv", index=False)
        print("train: {}, test: {}".format(len(train), len(test)))
        agree = (train["llm_label"] == train["gold_label"]).mean()
        print("LLM label agrees with the coder on {:.1%} of the training sentences".format(agree))
    elif what == "silver":
        # LLM-labeled sentences that were not audited, as extra training data for the encoders
        df = pd.read_csv(out / "llm_labels.csv", dtype=str, keep_default_na=False)
        sheet = pd.read_csv(sheet_file, dtype=str, keep_default_na=False)
        df = df[df["llm_label"].isin(global_options.LABELS) & ~df[global_options.SENT_COL].isin(sheet[global_options.SENT_COL])]
        for c in global_options.COMPONENTS:
            df["p_" + c] = df["llm_label"].map(lambda l: float(c in l.split("+")))
        cols = [global_options.SENT_COL, global_options.FQTR_COL, global_options.CQTR_COL, "llm_label"] + ["p_" + c for c in global_options.COMPONENTS]
        df[cols].to_csv(out / "silver.csv", index=False)
        print("{} silver sentences saved to {}".format(len(df), out / "silver.csv"))
    else:
        sys.exit("usage: python 03_build_gold.py sample|split|silver")
