"""Fine-tune a transformer encoder on the gold set.

Labels are decomposed into four binary components (Pa, P, F, s) and the encoder is
trained as a multi-label classifier. A compound label such as Pa+F is put back together
from the components at prediction time. Every encoder member of the ensemble is trained
with this script; they differ only in backbone and loss (see global_options.ENCODERS):

    asl            asymmetric loss on the four components
    coverage_mse   asl plus a squared-gap penalty when one gold component fires and another lags
    coverage_bce   asl plus a BCE pull-up on the lagging gold component

If data/output/silver.csv exists (03_build_gold.py silver), its LLM-labeled sentences are
added to the gold training set with a BCE loss on their component targets, weighted by
SILVER_WEIGHT; the gold rows keep the loss above. A validation split (VAL_SIZE of the gold
and silver rows together) is scored at a fixed step interval; the weights of the checkpoints
with the highest validation macro-F1 are averaged into the final model (the interval and
how many checkpoints are set per member in global_options.ENCODERS).

    python 04_train_encoder.py deberta_asl
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from safetensors.torch import load_file
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

import global_options

COMPONENTS = global_options.COMPONENTS


def label_to_binary(label):
    """'Pa+F' -> [1, 0, 1, 0]"""
    parts = set(str(label).split("+"))
    return [1.0 if c in parts else 0.0 for c in COMPONENTS]


def probs_to_label(p):
    """[0.9, 0.1, 0.8, 0.0] -> 'Pa+F': the components with probability >= 0.5, or the highest
    one if none reaches 0.5."""
    parts = [c for c, v in zip(COMPONENTS, p) if v >= 0.5] or [COMPONENTS[int(np.argmax(p))]]
    return "+".join(parts)


def build_text(sentence, fqtr, cqtr):
    """Input string seen by the encoder. The two focal quarters are prepended so the model
    can resolve explicit time anchors.
    """
    return "fiscal_quarter={} calendar_quarter={} sentence={}".format(fqtr, cqtr, sentence)


def asl_loss(logits, y, gamma_neg=global_options.ASL_GAMMA_NEG, clip=global_options.ASL_CLIP):
    """Asymmetric loss (Ridnik et al. 2021). Down-weights easy negatives, which dominate
    because most sentences carry only one or two components.
    """
    p = torch.sigmoid(logits)
    p_neg = (p - clip).clamp(min=0)
    pos = y * torch.log(p.clamp(min=1e-8))
    neg = (1 - y) * torch.log((1 - p_neg).clamp(min=1e-8)) * p_neg ** gamma_neg
    return -(pos + neg).sum(1).mean()


def coverage_loss(logits, y, kind, trigger=0.5):
    """Penalty for compound sentences where one gold component is predicted and another
    is missed. Only rows with two or more gold components contribute.
    """
    p = torch.sigmoid(logits)
    multi = y.sum(1) >= 2
    if multi.sum() == 0:
        return logits.new_zeros(())
    p, y = p[multi], y[multi]
    leader = ((p >= trigger) & (y == 1)).any(1, keepdim=True)
    lag = (p < trigger) & (y == 1) & leader
    n_lag = max(lag.sum().item(), 1)
    if kind == "coverage_mse":
        top = (p * y).max(1, keepdim=True).values
        return (((top - p) ** 2) * lag).sum() / n_lag
    else:
        return (F.binary_cross_entropy(p, torch.ones_like(p), reduction="none") * lag).sum() / n_lag


class MultiLabelTrainer(Trainer):
    def __init__(self, *args, loss_kind="asl", **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_kind = loss_kind

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        y = inputs.pop("labels")
        silver = inputs.pop("is_silver", None)
        logits = model(**inputs).logits
        gold = torch.ones(len(y), dtype=torch.bool, device=y.device) if silver is None else silver == 0
        loss = logits.new_zeros(())
        if gold.any():
            loss = asl_loss(logits[gold], y[gold])
            if self.loss_kind != "asl":
                loss = loss + global_options.COVERAGE_WEIGHT * coverage_loss(logits[gold], y[gold], self.loss_kind)
        if (~gold).any():
            loss = loss + global_options.SILVER_WEIGHT * F.binary_cross_entropy_with_logits(logits[~gold], y[~gold])
        return (loss, {"logits": logits}) if return_outputs else loss


def compute_metrics(eval_pred):
    """Validation metrics: macro-F1 over the four components and exact-match accuracy."""
    logits, y = eval_pred
    pred = (logits >= 0).astype(int)  # sigmoid(logit) >= 0.5
    return {"macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "accuracy": float((pred == y).all(1).mean())}


def to_dataset(df, tokenizer):
    """Tokenize a dataframe with columns sentence, datafqtr, datacqtr and either gold_label
    (gold rows) or p_<component> targets (silver rows, marked by is_silver = 1)."""
    text = [build_text(s, f, c) for s, f, c in
            zip(df[global_options.SENT_COL], df[global_options.FQTR_COL], df[global_options.CQTR_COL])]
    enc = tokenizer(text, truncation=True, max_length=global_options.MAX_LEN)
    silver = df["is_silver"].values if "is_silver" in df else np.zeros(len(df), dtype=int)
    enc["labels"] = [list(map(float, r[["p_" + c for c in COMPONENTS]])) if s else label_to_binary(r["gold_label"])
                     for (_, r), s in zip(df.iterrows(), silver)]
    enc["is_silver"] = [int(s) for s in silver]
    return Dataset.from_dict(dict(enc))


@torch.no_grad()
def predict_probs(model, tokenizer, df, batch_size=256):
    """Component probabilities for every row of df, in order.

    Arguments:
        model -- a trained encoder (AutoModelForSequenceClassification)
        tokenizer -- its tokenizer
        df {pd.DataFrame} -- rows with sentence, datafqtr, datacqtr

    Returns:
        np.ndarray -- shape (len(df), len(COMPONENTS))
    """
    model = model.eval()
    if hasattr(model, "module"):  # unwrap DataParallel
        model = model.module
    text = [build_text(s, f, c) for s, f, c in
            zip(df[global_options.SENT_COL], df[global_options.FQTR_COL], df[global_options.CQTR_COL])]
    out = []
    for i in range(0, len(text), batch_size):
        enc = tokenizer(text[i:i + batch_size], truncation=True, max_length=global_options.MAX_LEN,
                        padding=True, return_tensors="pt").to(model.device)
        out.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
    return np.vstack(out)


def load_training_data():
    """Hold out VAL_SIZE of the gold and optional silver rows for validation,
    then oversample rare gold classes in the training split only.
    """
    out = Path(global_options.OUTPUT_FOLDER)
    df = pd.read_csv(out / "gold" / "gold_train.csv")
    gold_sents = set(df[global_options.SENT_COL]) | set(pd.read_csv(out / "gold" / "gold_test.csv")[global_options.SENT_COL])
    df["is_silver"] = 0
    silver_file = out / "silver.csv"
    if silver_file.exists() and global_options.SILVER_WEIGHT > 0:
        silver = pd.read_csv(silver_file)
        silver = silver[~silver[global_options.SENT_COL].isin(gold_sents)].copy()
        silver["is_silver"] = 1
        print("silver: {} LLM-labeled sentences added with weight {}".format(len(silver), global_options.SILVER_WEIGHT))
        df = pd.concat([df, silver])
    label = df["gold_label"].fillna(df["llm_label"]) if "llm_label" in df else df["gold_label"]
    strat = label if label.value_counts().min() >= 2 else None
    train, val = train_test_split(df, test_size=global_options.VAL_SIZE, stratify=strat, random_state=global_options.SEED)
    for lab, k in global_options.OVERSAMPLE.items():
        rows = train[(train["is_silver"] == 0) & (train["gold_label"] == lab)]
        train = pd.concat([train] + [rows] * (k - 1), ignore_index=True)
    return train, val


def average_checkpoints(trainer, top_k):
    """Load the top_k checkpoints by validation macro-F1 that are still on disk and set the
    model weights to their average. Returns the steps that were averaged.
    """
    evals = [h for h in trainer.state.log_history if "eval_macro_f1" in h]
    evals.sort(key=lambda h: -h["eval_macro_f1"])
    files = [Path(trainer.args.output_dir, "checkpoint-{}".format(h["step"]), "model.safetensors") for h in evals]
    files = [f for f in files if f.exists()][:top_k]
    if not files:
        return []
    states = [load_file(str(f)) for f in files]
    target = trainer.model.state_dict()
    trainer.model.load_state_dict({k: (sum(s[k].float() for s in states) / len(states)).to(target[k].dtype) for k in target})
    return [int(f.parent.name.split("-")[1]) for f in files]


if __name__ == "__main__":
    member = sys.argv[1] if len(sys.argv) > 1 else "deberta_asl"
    backbone, loss_kind, top_k, every = global_options.ENCODERS[member]
    out_dir = Path(global_options.MODEL_FOLDER, member)
    print("Training {} ({}, loss={})".format(member, backbone, loss_kind))

    train, val = load_training_data()
    test = pd.read_csv(Path(global_options.OUTPUT_FOLDER, "gold", "gold_test.csv"))
    print("train: {} rows, validation: {} rows, test: {} rows".format(len(train), len(val), len(test)))

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=len(COMPONENTS), problem_type="multi_label_classification")
    eval_steps = min(every, max(1, len(train) // (global_options.BATCH_SIZE * global_options.GRAD_ACCUM)))
    args = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=global_options.EPOCHS, learning_rate=global_options.LR,
        per_device_train_batch_size=global_options.BATCH_SIZE, gradient_accumulation_steps=global_options.GRAD_ACCUM,
        per_device_eval_batch_size=64, warmup_ratio=0.06, lr_scheduler_type="cosine", weight_decay=0.01,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False),
        eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=eval_steps,
        save_total_limit=top_k + 1, load_best_model_at_end=True, metric_for_best_model="macro_f1",
        logging_steps=50, seed=global_options.SEED, report_to=[], remove_unused_columns=False)
    trainer = MultiLabelTrainer(model=model, args=args, train_dataset=to_dataset(train, tokenizer),
                                eval_dataset=to_dataset(val, tokenizer), compute_metrics=compute_metrics,
                                data_collator=DataCollatorWithPadding(tokenizer), loss_kind=loss_kind)
    trainer.train()
    averaged = average_checkpoints(trainer, top_k)
    print("averaged checkpoints at steps {}".format(averaged))
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    json.dump(trainer.state.log_history, open(out_dir / "log_history.json", "w"), indent=2)
    for ckpt in out_dir.glob("checkpoint-*"):
        shutil.rmtree(ckpt)

    # held-out predictions; the probabilities are the features for the stacker
    probs = predict_probs(trainer.model, tokenizer, test)
    test = test.copy()
    for i, c in enumerate(COMPONENTS):
        test["p_" + c] = probs[:, i]
    test["pred"] = [probs_to_label(r) for r in probs]
    acc = (test["pred"] == test["gold_label"]).mean()
    test.to_csv(out_dir / "test_probs.csv", index=False)
    json.dump({"member": member, "backbone": backbone, "loss": loss_kind, "averaged_steps": averaged,
               "test_accuracy": float(acc)}, open(out_dir / "metrics.json", "w"), indent=2)
    print("Test accuracy: {:.4f}".format(acc))
