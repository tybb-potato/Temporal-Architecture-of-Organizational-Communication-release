import glob
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import global_options


def _nc(pattern):
    """Make every group non-capturing (pandas warns on capturing groups in .str.contains)."""
    return re.sub(r"\((?!\?)", "(?:", pattern)


# --------------------------------------------------------------------------------------
# S1. Non-substantive sentence patterns
# --------------------------------------------------------------------------------------
MIN_WORDS = global_options.MIN_WORDS

BOILERPLATE = [
    r"forward[- ]looking statements?",
    r"safe harbor",
    r"factors that could cause actual results to differ",
    r"\bsec(?:urities)? filings?\b",
    r"\b10[- ]?k\b|\b10[- ]?q\b|\b8[- ]?k\b",
    r"non[- ]gaap (?:financial )?measures?",
    r"pursuant to the private securities litigation reform act",
    r"registration statement|proxy statement|prospectus|form s-?4",
]

OPERATOR_PROCEDURAL = [
    r"^\s*(?:operator|moderator|coordinator)\b[:,]",
    r"(?:your|our|the) (?:next|first) question (?:comes|will come) from",
    r"if you would like to ask a question",
    r"\bpress \*?\d\b",
    r"please limit yourself to one question",
    r"conference call (?:is being|will be) recorded",
    r"listen[- ]only mode",
    r"\bq&?\s?a\b session",
    r"your line is open",
    r"please go ahead",
    r"(this )?concludes (our|today'?s) (prepared remarks|remarks|call|conference call)",
    r"thank you for joining us(?: today)?",
    r"(?:joining us|on the call|on the line) today",
]

# "let me turn it over to ..." -- speaker handoffs carry no temporal content
SPEAKER_HANDOFF = [
    r"\b(?:turn|hand|pass)(?:ing)? (?:it|the call|things|us|you) (?:back |over |it )?(?:to|to you)\b",
    r"\bi(?:'d| would) like to (turn|hand) (it|the call|things) over\b",
    r"\bwith that,? (let me|i(?:'ll| will))\b.*\b(turn|hand)\b",
]

NONSUBSTANTIVE_RE = re.compile(
    _nc("|".join(BOILERPLATE + OPERATOR_PROCEDURAL + SPEAKER_HANDOFF)), re.IGNORECASE
)

# --------------------------------------------------------------------------------------
# S2. Temporal locatability: time anchors and future-intent signals
# --------------------------------------------------------------------------------------
MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
          r"aug(?:ust)?|sep(?:t\.?|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
SEASONS = r"(?:spring|summer|fall|autumn|winter)"
ORDINALS = r"(?:first|second|third|fourth|1st|2nd|3rd|4th)"
NUMW = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
APOST = r"(?:'|’)"

TIME_ANCHORS = [
    # explicit quarters
    r"\bQ[1-4]\b",
    rf"\b{ORDINALS}\s+quarter\b",
    r"\b(?:this|the|last|prior)[-\s]?quarter\b",
    r"\b(?:during|for)\s+the\s+quarter\b",
    r"\bquarter[-\s]?to[-\s]?date\b|\bqtd\b",
    r"\bQ[1-4]\s*(?:to|through|thru|[-–])\s*Q[1-4]\b",
    # years and fiscal years
    r"\b(?:19|20)\d{2}\b",
    rf"\bfy(?:\s*|[-_])?(?:{APOST}\d{{2}}|(?:19|20)\d{{2}})\b",
    rf"\bfiscal\s*year\s*(?:{APOST}\d{{2}}|(?:19|20)\d{{2}})\b",
    # relative year references
    r"\b(?:last|prior|previous)[-\s]?year\b",
    r"\byear[-\s]?ago\b",
    rf"\b{NUMW}\s+years?\s+ago\b",
    r"\byear[-\s]?(?:over|to)[-\s]?year\b|\byoy\b",
    r"\bquarter[-\s]?over[-\s]?quarter\b|\bqoq\b",
    # months, seasons, generic units
    rf"\b{MONTHS}\b",
    rf"\b{SEASONS}\b",
    r"\b(?:week|day|month|quarter|year)s?\b",
    # relative windows
    rf"\b(?:last|past|previous)\s+{NUMW}\s+(?:year|quarter|month|week|day)s?\b",
    rf"\b(?:over|in)\s+the\s+past\s+{NUMW}\s+(?:year|quarter|month|week|day)s?\b",
    rf"\bsince\s+(?:{MONTHS}|(?:19|20)\d{{2}}|{ORDINALS}\s+quarter|Q[1-4])\b",
]

FUTURE_SIGNALS = [
    r"\bwill\b", r"\bgoing\s+to\b", r"\bplan(?:s|ned|ning)?\s+to\b",
    r"\btarget(?:s|ed|ing)?\b", r"\bguid(?:ance|e|ed|es)\b", r"\boutlook\b",
    r"\bexpect(?:s|ed|ing|ation|ations)?\b",
    r"\banticipat(?:e|es|ed|ing|ion|ions)\b",
    r"\bforecast(?:s|ed|ing)?\b", r"\bproject(?:s|ed|ing|ion|ions)?\b",
    r"\bshould\b", r"\bcould\b", r"\bwould\b",
]

TIME_RE = re.compile(_nc("|".join(TIME_ANCHORS)), re.IGNORECASE)
FUTURE_RE = re.compile(_nc("|".join(FUTURE_SIGNALS)), re.IGNORECASE)

# --------------------------------------------------------------------------------------
# S3. Administrative sentences whose only future marker is procedural
# --------------------------------------------------------------------------------------
ADMIN_CONTEXT = [
    r"q&a|question[- ]and[- ]answer|questions and answers",
    r"open (?:up )?the (?:call|line|lines) for questions",
    r"take questions|begin the question and answer",
    r"operator|your line is open|please go ahead",
    r"prepared remarks|concludes (our|today'?s)",
    r"forward[- ]looking statements|safe harbor",
    r"replay|webcast|archived|post the (?:slides|presentation|materials)",
    r"press release|earnings release|slide presentation",
    r"record date|filed with the sec|sec filings",
    r"thank you for joining",
]
ADMIN_RE = re.compile(_nc("|".join(ADMIN_CONTEXT)), re.IGNORECASE)

# --------------------------------------------------------------------------------------
# S4. CEO speaker identification
# --------------------------------------------------------------------------------------
CEO_RE = re.compile(global_options.CEO_PATTERN, re.IGNORECASE)

# --------------------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------------------
TURN_COL = global_options.TURN_COL

# Protect decimals and common abbreviations so they do not trigger a sentence break.
_ABBREV = (r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Inc|Corp|Co|Ltd|LLC|LLP|Cos|Bros|"
           r"U\.S|U\.K|e\.g|i\.e|vs|etc|approx|No|Fig|Nos|Jan|Feb|Mar|Apr|Jun|"
           r"Jul|Aug|Sep|Sept|Oct|Nov|Dec)")
_PROTECT = [
    (re.compile(rf"\b{_ABBREV}\."), "\u0001"),        # abbreviation period
    (re.compile(r"(?<=\d)\.(?=\d)"), "\u0002"),        # decimal point
]
_SENT_END = re.compile(r"(?<=[.!?])[\"\u2019\u201d)\]]*\s+")


def split_sentences(text):
    """Split a speaker turn into sentences. Periods in decimals and common abbreviations
    are protected first so they do not end a sentence.

    Arguments:
        text {str} -- text of one speaker turn

    Returns:
        [str] -- list of sentences
    """
    if not isinstance(text, str) or not text.strip():
        return []
    t = text
    for rx, tok in _PROTECT:
        t = rx.sub(lambda m: m.group(0).replace(".", tok), t)
    t = re.sub(r"(?<=[.!?])(?=[A-Z])", " ", t)          # fix missing space after . ! ?
    parts = _SENT_END.split(t)
    out = []
    for s in parts:
        s = s.replace("\u0001", ".").replace("\u0002", ".").strip()
        if s:
            out.append(s)
    return out


def explode_to_sentences(df):
    """Expand a turn-level dataframe to one row per sentence. Only the columns listed in
    global_options.INPUT_COLS are kept.
    """
    missing = [c for c in global_options.INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing columns {missing}; got {list(df.columns)}")
    df = df[global_options.INPUT_COLS].copy()
    df["sentence"] = df[TURN_COL].map(split_sentences)
    df = df.drop(columns=[TURN_COL]).explode("sentence", ignore_index=True)
    return df[df["sentence"].notna()]


# --------------------------------------------------------------------------------------
# Selection rules
# --------------------------------------------------------------------------------------
def s1_substantive(s):
    """Keep sentences that are long enough and not boilerplate/procedural/handoff."""
    long_enough = s.str.split().str.len().fillna(0) >= MIN_WORDS
    return long_enough & ~s.str.contains(NONSUBSTANTIVE_RE, na=False)


def s2_temporally_locatable(s):
    """Keep sentences carrying a time anchor or a future-intent signal."""
    return s.str.contains(TIME_RE, na=False) | s.str.contains(FUTURE_RE, na=False)


def s3_not_admin_will(s):
    """Drop sentences whose only future marker is administrative ('will' + admin context)."""
    return ~(s.str.contains(r"\bwill\b", case=False, na=False)
             & s.str.contains(ADMIN_RE, na=False))


def s4_is_ceo(df):
    """Keep sentences spoken by the CEO, identified from title and role fields."""
    title_text = df[global_options.TITLE_COLS].fillna("").astype(str).agg(" ".join, axis=1)
    return title_text.str.contains(CEO_RE, na=False)


def select(df):
    """Apply the four selection rules in order.

    Arguments:
        df {pd.DataFrame} -- turn-level transcript table

    Returns:
        pd.DataFrame, dict -- selected sentences, and the count after each rule
    """
    df = explode_to_sentences(df)
    df["sentence"] = df["sentence"].astype(str)
    counts = {"split": len(df)}
    df = df[s1_substantive(df["sentence"])]
    counts["S1 substantive"] = len(df)
    df = df[s2_temporally_locatable(df["sentence"])]
    counts["S2 temporally locatable"] = len(df)
    df = df[s3_not_admin_will(df["sentence"])]
    counts["S3 not administrative"] = len(df)
    df = df[s4_is_ceo(df)]
    counts["S4 CEO"] = len(df)
    return df.reset_index(drop=True), counts


if __name__ == "__main__":
    files = sorted(glob.glob(global_options.INPUT_GLOB))
    assert len(files) > 0, "no input files found: " + global_options.INPUT_GLOB
    print("Selecting sentences from {} files".format(len(files)))
    frames = []
    total = {}
    for f in tqdm(files):
        if f.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(f)
        else:
            df = pd.read_csv(f, low_memory=False)
        df, counts = select(df)
        frames.append(df)
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
    out = pd.concat(frames, ignore_index=True)
    prev = None
    for k, v in total.items():
        removed = "" if prev is None else "  (-{})".format(prev - v)
        print("{:<28}{:>12,}{}".format(k, v, removed))
        prev = v
    print("calls: {}, firms: {}".format(out[global_options.DOC_COL].nunique(), out["gvkey"].nunique()))
    out_file = Path(global_options.OUTPUT_FOLDER, "ceo_sentences.csv")
    out.to_csv(out_file, index=False)
    print("Saved to", out_file)
