import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

import global_options

ALLOWED = set(global_options.LABELS)
CANONICAL = {l.lower(): l for l in global_options.LABELS}

SYSTEM_PROMPT = (
    # ---- Task and labels
    "You are an equity analyst specializing in earnings-call analysis.\n"
    "Label EACH sentence with EXACTLY ONE of: Pa, P, F, Pa+P, Pa+F, P+F, Pa+P+F, s.\n"
    "Rules:\n"
    "- One label only\n"
    "- If mixing time frames -> use the linked label\n"
    "- No explanation, just the label\n"
    "- Pa = Past (before the focal quarter)\n"
    "- P = Present (the just-ended quarter of the earnings call)\n"
    "- F = Future\n"
    "- s = Summary/meta sentence that describes the structure/content of the call or legal boilerplate,\n"
    "      rather than reporting actual performance or concrete plans.\n"
    "\n"

    # ---- 1. Reference points: datafqtr for quarter/year words, datacqtr for month names
    "IMPORTANT (Calendar month mapping vs. fiscal quarter anchors):\n"
    "- You will be given TWO focal quarters:\n"
    "  (A) Focal fiscal quarter (datafqtr) for quarter-words like Q1/Q2/\"in the quarter\": YYYYQ#\n"
    "  (B) Focal calendar quarter (datacqtr) ONLY for interpreting MONTH names: YYYYQ#\n"
    "- Use datacqtr ONLY to interpret MONTH names (\"Jan\", \"February\", etc.).\n"
    "- Month-to-quarter mapping is calendar-based: Jan–Mar=Q1, Apr–Jun=Q2, Jul–Sep=Q3, Oct–Dec=Q4.\n"
    "- Decide Pa vs P vs F for MONTH anchors by comparing the inferred (year, calendar quarter) to datacqtr.\n"
    "\n"
    "OVERRIDE (Use datafqtr for all non-month time reasoning):\n"
    "- If the sentence mentions ANY quarter/relative-quarter language (e.g., \"Q1\", \"first/second/third/fourth quarter\",\n"
    "early Q2\", \"end of Q3\"), interpret it using datafqtr.\n"
    "- If the sentence mentions ANY year/fiscal-year language (e.g., \"2017\", \"in 2020\", \"FY2021\", \"fiscal year 2022\"), interpret it using datafqtr as the focal point.\n"
    "- Comparison must be based on the (YEAR, QUARTER) combination when available:\n"
    "  • If an anchor is earlier than datafqtr in (year, quarter) order -> Past (Pa)\n"
    "  • If an anchor matches datafqtr (same year and quarter) -> P\n"
    "  • If an anchor is later than datafqtr in (year, quarter) order -> Future (F)\n"
    "- If a sentence contains BOTH quarter-words and month-names, prioritize the quarter/year interpretation via datafqtr;\n"
    "  use datacqtr month mapping only if the month clearly refers to a different period than the quarter reference.\n"
    "\n"

    # ---- 2. Present
    "Definition of Present:\n"
    "- First, determine the focal quarter.\n"
    "- If a sentence explicitly anchors to the focal quarter, classify as P.\n"
    "  Examples: \"in the quarter,\" \"this quarter,\" \"current quarter,\"\n"
    "  \"during the quarter,\" \"for the quarter,\" \"Q{quarter} of {year},\"\n"
    "- Present-tense 'current situation':\n"
    "- Treat present-tense statements about the business situation at the time of the call (e.g., \"now\", \"currently\", \"today\") as referring to the focal reporting period -> include P.\n"
    "- Add Pa and/or F if the same sentence also contains clear Past anchors or Future intent.\n"
    "\n"
    

    # ---- 3. When Present also reaches back: Pa+P
    "Comparison rule:\n"
    "- If the same sentence also compares the focal quarter with a prior period\n"
    "  (\"from last year,\" \"vs last year,\" \"YoY/y/y,\" \"year over year,\"\n"
    "   \"compared to last year/last quarter\"), upgrade to Pa+P.\n"
    "- If a sentence reports performance (growth, revenue, margin, results) without explicitly stating a time period,\n"
    "  and it includes a Past comparator (\"over last year,\" \"vs last year,\" \"year over year,\" \"YoY/y/y,\"\n"
    "  \"compared to last year/last quarter\"), treat the reported performance as referring to the focal quarter\n"
    "  and upgrade the label to Pa+P.\n"
    "\n"
    "Implicit change-words rule (Pa+P):\n"
    "- If a sentence explicitly anchors to the focal quarter or a specific quarter\n"
    "  (e.g., \"in the quarter,\" \"this quarter,\" \"for the quarter,\" \"in the first quarter\"),\n"
    "  AND it uses change words such as \"increased,\" \"decreased,\" \"declined,\" \"improved,\" \"was up,\" \"was down,\"\n"
    "  or phrases like \"up X%\" / \"down X%\" / \"up by\" / \"down by\" without explicitly stating what period it is\n"
    "  compared to, treat this as an implicit comparison between the focal quarter and the immediately\n"
    "  preceding period (the last quarter) and label it Pa+P.\n"
    "- Examples that should be labeled Pa+P under this rule include:\n"
    "  \"Cash flow from operating activities was up by approximately 200 million in the quarter.\"\n"
    "  \"Excluding net special charges and fuel our mainline cost per ASM was $0.0949 in the first quarter, up 5.8%.\"\n"
    "Implicit % change without explicit anchor (Pa+P):\n"
    "- If a sentence reports a realized performance number (e.g., revenue, earnings, margin)\n"
    "  AND includes a percent/delta change phrase such as \"up X%\" / \"down X%\" / \"increased X%\" / \"decreased X%\"\n"
    "  BUT does not explicitly state what period it is compared to,\n"
    "  treat the change as an implicit comparison to a prior period and label it Pa+P.\n"
    "- This applies even if the sentence does not explicitly say \"in the quarter\".\n"
    "- Example: \"Revenues were $1.28 billion, up 10%.\" -> Pa+P\n"
    "\n"
    "Aggregated period rule (Pa+P):\n"
    "- Some sentences describe performance over an aggregated period that includes BOTH:\n"
    "  (i) the just-ended focal quarter (P), AND\n"
    "  (ii) earlier time within the SAME fiscal year (Past),\n"
    "  using phrases such as \"year-to-date / year to date\", \"so far this year\", \"to date\", \"for the year\", \"for the full year\",\n"
    "  or mixing quarter numbers with year totals.\n"
    "\n"
    "- Label as Pa+P ONLY when the aggregated period truly includes time BEFORE the focal quarter within the same fiscal year.\n"
    "  Example: If the focal fiscal quarter is Q3 or Q4, then \"year-to-date\" necessarily includes earlier quarters -> Pa+P.\n"
    "\n"
    "Present perfect / ongoing-to-now rule (Pa+P, with implicit Future):\n"
    "- If a sentence uses present perfect or ongoing duration language (e.g., \"we've...\", \"has/have been...\",\n"
    "  \"to-date/to date\", \"so far\", \"for X years/quarters\", \"since ...\") and it implies an action or trend continuing up to the reporting period,\n"
    "  label it Pa+P (not Pa), even if it does not explicitly say \"in the quarter\".\n"
    "- OVERRIDE: If the sentence explicitly anchors the action/result to the focal quarter (\"this quarter\", \"in the quarter\", \"during the quarter\", \"for the quarter\", \"Q{quarter} {year}\"),\n"
    "  label it P unless it ALSO contains future elements below.\n"
    "- Add Future (upgrade to P+F or Pa+P+F) whenever the sentence includes any explicit or implicit signs.\n"
    "\n"

    # ---- 4. Future
    "F = Future if the sentence clearly refers to forward-looking statements,\n"
    "plans, guidance, projections, expectations, targets, future quarters, or outlook.\n"
    "- Important: Do NOT classify a sentence as Future (F) solely because it contains\n"
    "  words like \"project,\" \"projects,\" \"pipeline,\" \"initiative,\" \"platform,\"\n"
    "  \"program,\" or similar business nouns. Only assign F when the sentence\n"
    "  meaningfully expresses a forward-looking plan, intention, guidance,\n"
    "  expectation, or future action. \n"
    "\n"
   "Past-sense future verbs:\n"
    "- Phrases like \"didn't anticipate,\" \"met our expectation,\" \"as expected,\"\n"
    "  \"we thought,\" \"we anticipated\" are P unless they also compare to Past,\n"
    "  in which case label Pa+P.\n"
    "\n"

    # ---- 5. Summary / meta
    "Summary/meta sentences (s):\n"
    "- Use s when the main purpose of the sentence is to describe what will be covered on the call,\n"
    "  the speaking order, or generic forward-looking disclaimer language, even if time anchors appear.\n"
    "  Examples:\n"
    "  - \"Bob will then review the financial results for the quarter and provide an update on our\n"
    "     corporate goal of operational excellence.\"\n"
    "  - \"These statements relate to future events and expectations, and involve unknown risks\n"
    "     and uncertainties.\"\n"
    "  - \"Let me give you a few overall perspectives on the quarter for Agilent, review the performance\n"
    "     of our two business segments, and conclude with some thoughts about second quarter and\n"
    "     full year 2007 guidance.\"\n"
    "- If a sentence mixes call-structure language with substantial performance or guidance content,\n"
    "  prioritize Pa/P/F-based labels over s.\n"
    "\n"

    # ---- Combined labels and output format
    "If a sentence mixes Present and Future -> P+F.\n"
    "If it mixes Past and Future -> Pa+F.\n"
    "If it mixes all three -> Pa+P+F.\n"
    "\n"
    "When you answer, return a single JSON object with two keys:\n"
    "  \"label\": one of [Pa, P, F, Pa+P, Pa+F, P+F, Pa+P+F, s]\n"
    "  \"evidence\": a short phrase or phrases copied from the sentence that justify the label\n"
    "Example:\n"
    "{\"label\": \"Pa+P\", \"evidence\": \"in the quarter; up 5.8%\"}\n"
)


YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
YQ_YYYYQ_RE = re.compile(r"\b((?:19|20)\d{2})\s*Q([1-4])\b", re.IGNORECASE)
YQ_QYYYY_RE = re.compile(r"\bQ([1-4])\s*((?:19|20)\d{2})\b", re.IGNORECASE)
YQ_QWORD_YEAR_RE = re.compile(
    r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter(?:\s+of)?\s+((?:19|20)\d{2})\b",
    re.IGNORECASE
)
Q_ONLY_RE = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
QWORD_ONLY_RE = re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\b", re.IGNORECASE)
REL_QUARTER_RE = re.compile(r"\b(in|for|during)\s+the\s+quarter\b|\bthis\s+quarter\b|\bcurrent\s+quarter\b", re.IGNORECASE)
MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t\.?|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE
)


def normalize_label(raw):
    """Return one of the eight labels (case, spaces and surrounding punctuation ignored),
    or an empty string if the reply is anything else."""
    if not isinstance(raw, str):
        return ""
    lab = raw.strip().strip(".,:;()[]\"'").replace(" ", "")
    return CANONICAL.get(lab.lower(), "")


def split_on_semicolons(sentence):
    parts = [p.strip(" \t,") for p in (sentence or "").split(";")]
    return [p for p in parts if p]


def union_labels(labels):
    labels = [l for l in labels if l in ALLOWED and l]
    if not labels:
        return ""
    if all(l == "s" for l in labels):
        return "s"

    has_pa = any("Pa" in l.split("+") for l in labels if l != "s")
    has_p = any("P" in l.split("+") for l in labels if l != "s")
    has_f = any("F" in l.split("+") for l in labels if l != "s")

    out_parts = []
    if has_pa: out_parts.append("Pa")
    if has_p: out_parts.append("P")
    if has_f: out_parts.append("F")

    out = "+".join(out_parts) if out_parts else "s"
    return out if out in ALLOWED else ""


def detect_anchors(sentence):
    s = sentence or ""

    yq = []
    for m in YQ_YYYYQ_RE.finditer(s):
        yq.append(m.group(0))
    for m in YQ_QYYYY_RE.finditer(s):
        yq.append(m.group(0))
    for m in YQ_QWORD_YEAR_RE.finditer(s):
        yq.append(m.group(0))

    years = sorted({m.group(0) for m in YEAR_RE.finditer(s)})

    q_only = []
    for m in Q_ONLY_RE.finditer(s):
        q_only.append(m.group(0))
    for m in QWORD_ONLY_RE.finditer(s):
        q_only.append(m.group(0))
    for m in REL_QUARTER_RE.finditer(s):
        q_only.append(m.group(0))

    months = sorted({m.group(0) for m in MONTH_RE.finditer(s)})

    def uniq(lst):
        out = []
        seen = set()
        for x in lst:
            k = x.lower()
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out

    return {
        "explicit_year_quarter": uniq(yq),
        "years": years,
        "quarters": uniq(q_only),
        "months": uniq(months),
    }


def build_user_input(datafqtr, datacqtr, sentence):
    """Assemble the user message. Time anchors found by regex are listed so the model
    does not have to search for them.
    """
    anchors = detect_anchors(sentence)
    return (
        f"Focal fiscal quarter (datafqtr): {datafqtr}\n"
        f"Focal calendar quarter for MONTH names only (datacqtr): {datacqtr}\n"
        f"Detected anchors (no inference):\n"
        f"- explicit year+quarter: {anchors['explicit_year_quarter']}\n"
        f"- years: {anchors['years']}\n"
        f"- quarters/quarter-phrases: {anchors['quarters']}\n"
        f"- month names: {anchors['months']}\n"
        f"Sentence: {sentence}\n"
        f"Return a JSON object with keys 'label' and 'evidence'."
    )


def classify_api_once(datafqtr, datacqtr, sentence):
    """Call the API for one sentence.

    Returns:
        (str, str) -- label (empty string if the model did not return a valid label), evidence
    """
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + global_options.LLM_API_KEY}
    payload = {
        "model": global_options.LLM_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": build_user_input(datafqtr, datacqtr, sentence)}],
        "temperature": global_options.LLM_TEMPERATURE,
    }
    sleep = 1.5
    for attempt in range(1, global_options.LLM_MAX_RETRIES + 1):
        try:
            r = requests.post(global_options.LLM_BASE_URL, headers=headers, json=payload, timeout=60)
            if r.status_code != 200:
                print(f"API error (attempt {attempt}): HTTP {r.status_code}")
                time.sleep(sleep)
                sleep *= 2
                continue
            text = r.json()["choices"][0]["message"]["content"].strip()
            # some models wrap the json in a code block
            if text.startswith("```"):
                text = "\n".join(l for l in text.split("\n") if not l.startswith("```")).strip()
            label_raw, evidence = text, ""
            try:
                data = json.loads(text)
                label_raw = data.get("label", "")
                evidence = data.get("evidence", "")
            except json.JSONDecodeError:
                pass
            out = normalize_label(label_raw)
            if out in ALLOWED:
                return out, evidence
            print(f"Bad label: raw='{label_raw}'")
            return "", evidence
        except Exception as e:
            print(f"API error (attempt {attempt}): {type(e).__name__}: {e}")
            time.sleep(sleep)
            sleep *= 2
    return "", ""


def classify_one(datafqtr, datacqtr, sentence, allow_split=True):
    """Label one sentence. Sentences joined by semicolons are labeled clause by clause and
    the labels are merged (a clause about the past and a clause about the future give Pa+F);
    if any clause gets no valid label, the sentence gets none either.
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return "", ""
    if allow_split and ";" in sentence:
        clauses = split_on_semicolons(sentence)
        if len(clauses) >= 2:
            labs, evs = [], []
            for c in clauses:
                lab, ev = classify_one(datafqtr, datacqtr, c, allow_split=False)
                if not lab:
                    return "", ""
                labs.append(lab)
                evs.append(ev)
            return union_labels(labs), "; ".join([e for e in evs if e][:3])
    return classify_api_once(datafqtr, datacqtr, sentence)


if __name__ == "__main__":
    assert global_options.LLM_API_KEY, "set the LLM_API_KEY environment variable"
    in_file = Path(global_options.OUTPUT_FOLDER, "ceo_sentences.csv")
    out_file = Path(global_options.OUTPUT_FOLDER, "llm_labels.csv")
    df = pd.read_csv(in_file, dtype=str, keep_default_na=False)
    if len(df) > global_options.N_LLM:
        df = df.sample(global_options.N_LLM, random_state=global_options.SEED).reset_index(drop=True)
    print(f"Loaded {len(df)} sentences")
    # resume from the checkpoint if there is one; rows without a valid label are (re)tried
    labels, evidences = [""] * len(df), [""] * len(df)
    if out_file.exists():
        done = pd.read_csv(out_file, dtype=str, keep_default_na=False)
        key = [global_options.SENT_COL, global_options.FQTR_COL, global_options.CQTR_COL]
        assert done[key].equals(df[key]), "checkpoint does not match the input; delete {} to start over".format(out_file)
        labels, evidences = list(done["llm_label"]), list(done["llm_evidence"])
    todo = [i for i, lab in enumerate(labels) if lab not in ALLOWED]
    print(f"{len(todo)} sentences to label")
    for k, i in enumerate(todo, 1):
        row = df.iloc[i]
        labels[i], evidences[i] = classify_one(row[global_options.FQTR_COL], row[global_options.CQTR_COL], row[global_options.SENT_COL])
        if k % global_options.LLM_SAVE_INTERVAL == 0 or k == len(todo):
            part = df.copy()
            part["llm_label"] = labels
            part["llm_evidence"] = evidences
            part.to_csv(out_file, index=False)
            print(f"{k}/{len(todo)} rows, checkpoint saved")
    print("Done. {} sentences without a valid label; run again to retry them. Saved to {}".format(
        sum(lab not in ALLOWED for lab in labels), out_file))
