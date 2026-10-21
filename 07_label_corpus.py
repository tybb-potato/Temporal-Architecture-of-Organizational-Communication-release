"""Label every selected sentence with the ensemble.

Runs each encoder member over the corpus for component probabilities, each LLM member
for a hard label, and applies the saved stacker. Members are read in the order of
global_options.ENCODERS followed by global_options.LLMS, the same order 06_stack_models.py used.

    python 07_label_corpus.py
"""
import importlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import global_options

# script names start with a digit, so they cannot be imported with `import`
predict_probs = importlib.import_module("04_train_encoder").predict_probs
_llm = importlib.import_module("05_train_llm")
generate_labels, load_base = _llm.generate_labels, _llm.load_base


def encoder_probs(model_dir, df):
    """Component probabilities from one saved encoder member."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return predict_probs(model, tokenizer, df)

if __name__ == "__main__":
    in_file = Path(global_options.OUTPUT_FOLDER, "ceo_sentences.csv")
    df = pd.read_csv(in_file)
    print("Loaded {} sentences".format(len(df)))

    blocks, names = [], []
    for member in global_options.ENCODERS:
        print("Running", member)
        blocks.append(encoder_probs(Path(global_options.MODEL_FOLDER, member), df))
        names += ["{}_{}".format(member, c) for c in global_options.COMPONENTS]
    for member, base_model in global_options.LLMS.items():
        print("Running", member)
        llm_dir = Path(global_options.MODEL_FOLDER, member)
        tokenizer = AutoTokenizer.from_pretrained(str(llm_dir))
        model = load_base(base_model, with_adapter=llm_dir).eval()
        preds = generate_labels(model, tokenizer, df, batch_size=16)
        blocks.append(np.array([[float(c in p.split("+")) for c in global_options.COMPONENTS] for p in preds]))
        names += ["{}_{}".format(member, c) for c in global_options.COMPONENTS]
        del model
        torch.cuda.empty_cache()
    X = np.hstack(blocks)

    stacker = joblib.load(Path(global_options.MODEL_FOLDER, "stacker.joblib"))
    assert names == stacker["features"], "members do not match the stacker"
    df["label"] = stacker["model"].predict(X)
    out_file = Path(global_options.OUTPUT_FOLDER, "ceo_sentences_labeled.csv")
    df.to_csv(out_file, index=False)
    print(df["label"].value_counts())
    print("Saved to", out_file)
