# Coding rules for temporal orientation

Each sentence gets exactly one label for the time period(s) it refers to. Periods are
judged relative to the focal quarter, i.e. the fiscal quarter the earnings call reports on.

Pa: past. Periods before the focal quarter.

P: present. The focal quarter itself ("this quarter", "in the quarter", "Q3") and the
situation at the time of the call ("now", "currently", "today").

F: future. Periods after the focal quarter: plans, guidance, expectations, targets, later
quarters or years.

Pa+P, Pa+F, P+F, Pa+P+F: the sentence refers to more than one period.

s: non-temporal. The sentence is about the structure of the call (who speaks next, what
will be covered), a speaker handoff, or legal boilerplate, even if it contains time words.

## How to resolve time references

1. Quarter and year words ("second quarter", "Q4", "2018", "fiscal 2022") are compared with
   the focal fiscal quarter (`datafqtr`). Earlier is Pa, the same is P, later is F.
2. Month names are compared with the focal calendar quarter (`datacqtr`), using
   Jan-Mar = Q1, Apr-Jun = Q2, Jul-Sep = Q3, Oct-Dec = Q4.
3. A result reported with a change but no comparison period ("revenue was $1.2 billion, up
   10%") compares the focal quarter with an earlier period: Pa+P.
4. Present perfect and "since"/"over the past X years" phrases describe something that
   started in the past and continues to the reporting period: Pa+P.
5. "expect", "plan", "guidance", "outlook", "will", "target" mark F only when the sentence
   states a forward-looking intention or expectation. Nouns like "project", "pipeline" or
   "program" do not by themselves make a sentence F.


## Examples (focal quarter 2019Q3)

Two years ago we launched the Falcon coating line. -- Pa

We currently have three new plants under construction. -- P

We expect to complete the first of them in the second quarter of 2020. -- F

Revenue in the third quarter was $412 million, up 9% from the same quarter last year. -- Pa+P

In 2017 we closed our Dallas plant, and we plan to reopen it in 2021. -- Pa+F

Given the momentum we saw in September, we are raising full-year guidance. -- P+F

Cash flow was $58 million in the quarter, up from $41 million a year ago, and we plan to return most of it in 2020. -- Pa+P+F

Now let me hand it over to Priya to walk through the financials. -- s

More examples are in `data/input/sample_gold.csv`.
