# Temporal Architecture of Organizational Communication

## Introduction

This repository contains the code for measuring temporal orientation in CEO earnings call
communication. Each CEO sentence is labeled with the time period(s) it
refers to: past (Pa), present (P), future (F), a combination of these (Pa+P, Pa+F, P+F,
Pa+P+F), or non-temporal (s). The sentence labels are then aggregated into a call-level
measure of temporal diversity.


## Requirement

Python 3.11 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

Steps 1 to 3, 6 and 8 run on a CPU. Steps 4, 5 and 7 need a GPU. The encoders
(DeBERTa-v3-large, RoBERTa-large) train on a 16G card; the two 32B instruction models in
4-bit need about 32G each. The backbones are downloaded from Hugging Face.

Step 2 labels the sentences with a large language model through its API. We used GPT-4.1
and gemini-3-pro-preview. Put the API key in the environment variable `LLM_API_KEY`; the
URL and model name are set in `global_options.py`.

To check the installation (CPU part only), run step 1 on the sample transcripts, then
step 3 with the sample LLM labels in place of the step 2 output:

```
python 01_select_sentences.py
cp data/input/sample_llm_labeled.csv data/output/llm_labels.csv
python 03_build_gold.py sample
```

## Data

The folder `data/input/` has three small example files. The sentences in them are made
up for illustration; no real transcript text is distributed here.

`sample_transcripts.csv`: the format of the cleaned transcripts, one row per speaker turn.
The columns that the code uses are

- `Document Name`: identifies the call
- `Speaker Name`, `Title`, `Role_Final`: the speaker should be the CEO
- `Comments`: the text of the turn
- `gvkey`: firm identifier
- `datafqtr`: the fiscal quarter the call reports on, in the form 2019Q3
- `datacqtr`: the calendar quarter

`sample_llm_labeled.csv`: the format of the step 2 output, with the LLM label
(`llm_label`) and its evidence phrase (`llm_evidence`).

`sample_gold.csv`: the format of the audited gold set (`sentence`, `datafqtr`,
`datacqtr`, `gold_label`, `coder_note`). The coding rules are in `resources/codebook.md`.

## Before running the code

All settings are in `global_options.py`. The ones to look at first:

- `INPUT_GLOB`: which transcript files to read. In this example, we use prepared remarks from earnings calls.
- `CEO_PATTERN`: regular expression matched against the speaker title and role.
- `LLM_BASE_URL`, `LLM_MODEL`: the large language model used for the labels.
- `N_LLM`, `N_AUDIT`, `TEST_SIZE`: how many sentences the LLM labels, how many of them go
  to the human coders, and what share of the coded sentences is held out for evaluation.
- `SILVER_WEIGHT`: loss weight of the LLM-labeled sentences that were not audited when
  training the encoders.
- `ENCODERS`, `LLMS`: the members of the ensemble. Their order is the feature order of
  the stacker, so do not change it between `06_stack_models.py` and `07_label_corpus.py`.

The LLM prompt is `SYSTEM_PROMPT` in `02_llm_label.py`. The same rules are written out for
the human coders in `resources/codebook.md`.

## Running the code

1. `python 01_select_sentences.py`

   Splits every turn into sentences and keeps a sentence if it (i) has at least five words
   and is not operator talk or a speaker handoff, (ii) contains a time anchor
   or a future-intent word, (iii) is not administrative ("a replay will be available"), and
   (iv) is spoken by the CEO. The number of sentences dropped by each rule is printed.
   Output: `data/output/ceo_sentences.csv`.

2. `python 02_llm_label.py`

   Sends a random sample of `N_LLM` selected sentences, each with its two focal quarters,
   to the LLM and stores the label and an evidence phrase. Output: `data/output/llm_labels.csv`.

3. `python 03_build_gold.py sample`, then `split`, then `silver`

   `sample` draws a random sample of the LLM-labeled sentences for human coding and
   writes it to `data/output/audit_sheet.csv` with an empty `gold_label` column for the
   coders to fill in. After coding, `split` divides the coded
   sentences into a training set (75%) and a held-out test set (25%), keeping the label
   proportions the same in both, and writes `data/output/gold/gold_train.csv` and
   `gold_test.csv`. `silver` writes the LLM-labeled sentences that were not audited to
   `data/output/silver.csv` as extra training data for the encoders.

4. `python 04_train_encoder.py <member>`, once for each encoder member:

   ```
   python 04_train_encoder.py deberta_asl
   python 04_train_encoder.py deberta_cov_mse
   python 04_train_encoder.py deberta_cov_bce
   python 04_train_encoder.py roberta
   ```

   The encoders are trained as multi-label classifiers over the four components Pa, P, F
   and s; a compound label is recovered from the components. The three DeBERTa runs use
   the same data and seed and differ only in the loss (see the docstring of
   `04_train_encoder.py`). A validation split of the training data, gold and silver
   together, is used to select the checkpoints.

5. `python 05_train_llm.py <member>`, once for each LLM member:

   ```
   python 05_train_llm.py qwen_lora
   python 05_train_llm.py deepseek_lora
   ```

   Fine-tunes each instruction model with QLoRA to generate the label as text. The checkpoint
   with the lowest loss on a validation split is kept.

6. `python 06_stack_models.py`

   Fits a logistic regression on the members' held-out outputs, prints each member's
   held-out accuracy and the stacker's out-of-fold accuracy, and saves
   `models/stacker.joblib`.

7. `python 07_label_corpus.py`

   Runs every member over `ceo_sentences.csv` and applies the stacker. Output:
   `data/output/ceo_sentences_labeled.csv`.

8. `python 08_aggregate_calls.py`

   Computes, for each call, the number of temporal sentences and temporal diversity (mean
   of 1 for one orientation, 2 for two, 3 for three). Output: `data/output/call_measures.csv`.
