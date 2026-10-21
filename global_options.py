"""Global options for the TAOC pipeline
"""
import os
from pathlib import Path
from typing import Dict, List

# Directory locations
DATA_FOLDER: str = "data/"
INPUT_FOLDER: str = os.path.join(DATA_FOLDER, "input")  # cleaned transcripts, one row per speaker turn
OUTPUT_FOLDER: str = os.path.join(DATA_FOLDER, "output")
MODEL_FOLDER: str = "models/"
INPUT_GLOB: str = os.path.join(INPUT_FOLDER, "*transcripts*.csv")  # which input files to read in 01_select_sentences.py

# Column names in the cleaned transcripts
DOC_COL: str = "Document Name"  # identifies the call
TURN_COL: str = "Comments"  # text of the speaker turn
TITLE_COLS: List[str] = ["Title", "Role_Final"]  # columns used to identify the CEO
SENT_COL: str = "sentence"  # created by 01_select_sentences.py
FQTR_COL: str = "datafqtr"  # focal fiscal quarter, YYYYQ#
CQTR_COL: str = "datacqtr"  # focal calendar quarter, YYYYQ#, used to interpret month names
CALL_ID: List[str] = [DOC_COL, "gvkey", "datafqtr"]  # keys that identify one earnings call
INPUT_COLS: List[str] = [DOC_COL, "Speaker Name"] + TITLE_COLS + [TURN_COL, "gvkey", FQTR_COL, CQTR_COL]

# Sentence selection
MIN_WORDS: int = 5  # sentences shorter than this are dropped
CEO_PATTERN: str = r"chief executive officer|chief executive|\bceo\b|principal executive officer"

# Labels
LABELS: List[str] = ["Pa", "P", "F", "Pa+P", "Pa+F", "P+F", "Pa+P+F", "s"]
COMPONENTS: List[str] = ["Pa", "P", "F", "s"]  # encoders predict these four; compound labels are recovered from them

# LLM labeling (02_llm_label.py): model API address, model name, key
LLM_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
LLM_MODEL: str = "gemini-3-pro-preview"
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
LLM_TEMPERATURE: float = 0
LLM_MAX_RETRIES: int = 5
LLM_SAVE_INTERVAL: int = 25  # write a checkpoint every N rows
N_LLM: int = 25750  # random sample of the selected sentences labeled by the LLM: N_AUDIT for the coders, the rest silver

# Human audit (03_build_gold.py)
N_AUDIT: int = 3750  # number of sentences sent to human coders
TEST_SIZE: float = 0.25  # share of the audited sentences held out for evaluation
SEED: int = 42

# Encoder training (04_train_encoder.py)
MAX_LEN: int = 256
EPOCHS: int = 6
LR: float = 1.5e-5
BATCH_SIZE: int = 8
GRAD_ACCUM: int = 4  # batches per update: BATCH_SIZE x GRAD_ACCUM = 32 sentences
OVERSAMPLE: Dict[str, int] = {"s": 3, "Pa": 3, "Pa+F": 3, "P+F": 3, "Pa+P+F": 3}  # repeat rare classes in training
SILVER_WEIGHT: float = 0.5  # loss weight of the LLM-labeled (silver) sentences in data/output/silver.csv; 0 = gold only
VAL_SIZE: float = 0.10  # share of the training data (gold and silver together) held out to pick the checkpoint
ASL_GAMMA_NEG: float = 4.0  # asymmetric loss
ASL_CLIP: float = 0.05
COVERAGE_WEIGHT: float = 1.0  # weight of the coverage penalty in the coverage_* variants

# Ensemble members. Encoders are trained by 04_train_encoder.py;
# LLMs by 05_train_llm.py. The order is the feature order of the stacker.
ENCODERS: Dict[str, tuple] = {
    # backbone, loss, best checkpoints averaged, evaluate every N steps
    "deberta_asl": ("microsoft/deberta-v3-large", "asl", 4, 100),
    "deberta_cov_mse": ("microsoft/deberta-v3-large", "coverage_mse", 4, 100),
    "deberta_cov_bce": ("microsoft/deberta-v3-large", "coverage_bce", 4, 100),
    "roberta": ("roberta-large", "asl", 1, 50),
}
LLMS: Dict[str, str] = {
    "qwen_lora": "Qwen/Qwen2.5-32B-Instruct",
    "deepseek_lora": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}
LORA_RANK: int = 16
LLM_EPOCHS: int = 3
LLM_LR: float = 1e-4
LLM_OVERSAMPLE: Dict[str, int] = {"s": 3, "Pa": 3, "Pa+F": 3, "P+F": 3, "Pa+P+F": 4}  # repeat rare classes in training
LLM_VAL_SIZE: float = 0.05  # share of the training data held out; the checkpoint with the lowest validation loss is kept
LLM_EVAL_STEPS: int = 100  # evaluate on the validation split every this many steps

# Stacker (06_stack_models.py)
STACK_FOLDS: int = 5
STACK_REPEATS: int = 30
STACK_C: float = 1.0

# Create directories if not exist
Path(INPUT_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER, "gold").mkdir(parents=True, exist_ok=True)
Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
