"""Fine-tune an instruction LLM with QLoRA as a generative labeler.

The model reads a prompt with the two focal quarters and the sentence and writes the
label. It is trained on the gold training set, with rare classes repeated (LLM_OVERSAMPLE),
with 4-bit base weights and LoRA adapters; a validation split (LLM_VAL_SIZE) is scored every
LLM_EVAL_STEPS steps and the checkpoint with the lowest validation loss is kept, then run on
the test set. Its output is a hard label; the stacker splits it into the four components.
Run once for each LLM member in global_options.LLMS:

    python 05_train_llm.py qwen_lora
    python 05_train_llm.py deepseek_lora
"""
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)

import global_options

SYSTEM = ("Label the earnings-call sentence by the time period(s) it refers to, relative to the focal "
          "fiscal quarter. Answer with exactly one of: " + ", ".join(global_options.LABELS) + ".")
LABEL_RE = re.compile(r"^\s*(Pa\+P\+F|Pa\+P|Pa\+F|P\+F|Pa|P|F|s)\b")


def messages(row, with_answer):
    m = [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Focal fiscal quarter: {}\nFocal calendar quarter: {}\nSentence: {}".format(
             row[global_options.FQTR_COL], row[global_options.CQTR_COL], row[global_options.SENT_COL])}]
    if with_answer:
        m.append({"role": "assistant", "content": row["gold_label"]})
    return m


def encode(df, tokenizer, max_len=384):
    """Tokenize prompt + answer; the prompt tokens are masked out of the loss."""
    ids, labels = [], []
    for r in df.to_dict("records"):
        prompt = tokenizer.apply_chat_template(messages(r, False), tokenize=False, add_generation_prompt=True)
        a_ids = tokenizer(r["gold_label"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][:max_len - len(a_ids)]
        ids.append(p_ids + a_ids)
        labels.append([-100] * len(p_ids) + a_ids)
    return Dataset.from_dict({"input_ids": ids, "labels": labels})


@torch.no_grad()
def generate_labels(model, tokenizer, df, batch_size=8):
    """Greedy decoding; the first label pattern in the output is taken as the prediction."""
    tokenizer.padding_side = "left"
    out = []
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i + batch_size].to_dict("records")
        prompts = [tokenizer.apply_chat_template(messages(r, False), tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=16, do_sample=False)
        for text in tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
            m = LABEL_RE.match(text.split("</think>")[-1])  # skip a reasoning block if the model opens one
            out.append(m.group(1) if m else "")  # an output that is not a label gets no vote
    return out


def load_base(base_model, with_adapter=None):
    """4-bit base model, optionally with a trained adapter on top."""
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb, device_map="auto")
    if with_adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(with_adapter))
    return model


if __name__ == "__main__":
    member = sys.argv[1] if len(sys.argv) > 1 else "qwen_lora"
    base_model = global_options.LLMS[member]
    out_dir = Path(global_options.MODEL_FOLDER, member)
    print("Training {} ({})".format(member, base_model))
    gold = Path(global_options.OUTPUT_FOLDER, "gold")
    df = pd.read_csv(gold / "gold_train.csv")
    test = pd.read_csv(gold / "gold_test.csv")
    strat = df["gold_label"] if df["gold_label"].value_counts().min() >= 2 else None
    train, val = train_test_split(df, test_size=global_options.LLM_VAL_SIZE, stratify=strat, random_state=global_options.SEED)
    for lab, k in global_options.LLM_OVERSAMPLE.items():
        train = pd.concat([train] + [train[train["gold_label"] == lab]] * (k - 1), ignore_index=True)
    print("train: {} rows, validation: {} rows, test: {} rows".format(len(train), len(val), len(test)))

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = prepare_model_for_kbit_training(load_base(base_model))
    r = global_options.LORA_RANK
    model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))

    eval_steps = min(global_options.LLM_EVAL_STEPS, max(1, len(train) // 16))  # 16 = batch x accumulation
    args = TrainingArguments(output_dir=str(out_dir), num_train_epochs=global_options.LLM_EPOCHS,
                             learning_rate=global_options.LLM_LR, warmup_ratio=0.05, lr_scheduler_type="cosine",
                             per_device_train_batch_size=4,
                             gradient_accumulation_steps=4, bf16=True, logging_steps=20, report_to=[],
                             eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=eval_steps,
                             save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss")
    trainer = Trainer(model=model, args=args, train_dataset=encode(train, tokenizer), eval_dataset=encode(val, tokenizer),
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True))
    trainer.train()
    print("best checkpoint: {}".format(trainer.state.best_model_checkpoint))
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    json.dump(trainer.state.log_history, open(out_dir / "log_history.json", "w"), indent=2)
    for ckpt in out_dir.glob("checkpoint-*"):
        shutil.rmtree(ckpt)

    model.eval()
    test = test.copy()
    test["pred"] = generate_labels(model, tokenizer, test)
    acc = (test["pred"] == test["gold_label"]).mean()
    print("{} outputs were not a label".format((test["pred"] == "").sum()))
    test.to_csv(out_dir / "test_preds.csv", index=False)
    json.dump({"member": member, "base": base_model, "best_checkpoint": trainer.state.best_model_checkpoint,
               "test_accuracy": float(acc)}, open(out_dir / "metrics.json", "w"), indent=2)
    print("Test accuracy: {:.4f}".format(acc))
