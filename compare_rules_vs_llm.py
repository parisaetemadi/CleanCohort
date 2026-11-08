"""Read the same notes twice - rules, then an AI model - and compare the answers.

Run:  python compare_rules_vs_llm.py          (needs ANTHROPIC_API_KEY)
      python compare_rules_vs_llm.py --rules-only   (no key; checks the script works)
"""

import argparse
import os

import pandas as pd

from cleancohort.pipeline import run_pipeline
from cleancohort.step0_make_fake_data import main as make_fake_data
from cleancohort.step4_read_notes import make_anthropic_caller
from cleancohort.vocab import CONDITIONS

PRICE_IN_PER_MTOK = 2.00
PRICE_OUT_PER_MTOK = 10.00


def make_counting_caller():
    """Same call as step4, plus a running total of tokens so we can price the run."""
    import anthropic

    client = anthropic.Anthropic()
    model = os.environ.get("CLEANCOHORT_LLM_MODEL", "claude-sonnet-5-5")
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "refusals": 0}

    def call(prompt: str) -> str:
        msg = client.messages.create(model=model, max_tokens=2000,
                                     output_config={"effort": "low"},
                                     messages=[{"role": "user", "content": prompt}])
        usage["calls"] += 1
        usage["input_tokens"] += msg.usage.input_tokens
        usage["output_tokens"] += msg.usage.output_tokens
        if msg.stop_reason == "refusal":
            usage["refusals"] += 1
        return "".join(b.text for b in msg.content if b.type == "text")

    return call, usage, model


def smoke_test_real_caller():
    """Prove make_anthropic_caller itself works against the live API, on one tiny prompt."""
    call = make_anthropic_caller()
    reply = call('Reply with exactly this JSON and nothing else: {"ok": true}')
    return reply.strip()


def accuracy_side_by_side(rules_val, llm_val) -> pd.DataFrame:
    keep = ["condition", "sensitivity", "PPV", "TP", "FP", "FN"]
    r = rules_val["conditions"].query("method == 'EHR + notes'")[keep]
    l = llm_val["conditions"].query("method == 'EHR + notes'")[keep]
    return r.merge(l, on="condition", suffixes=("_rules", "_llm"))


def notes_only_agreement(rules_ds, llm_ds, crosswalk, truth_path) -> pd.DataFrame:
    """Where the EHR codes missed a true case, which reader found it in the notes?"""
    truth = pd.read_csv(truth_path).merge(crosswalk, on="mrn")
    rows = []
    for c in CONDITIONS:
        r = rules_ds[["study_id", f"has_{c}", f"{c}_source"]].merge(truth[["study_id", c]], on="study_id")
        l = llm_ds[["study_id", f"has_{c}"]].merge(truth[["study_id", c]], on="study_id")
        in_ehr = r[f"{c}_source"].fillna("").str.contains("ehr")
        missed = set(r[r[c].astype(bool) & ~in_ehr]["study_id"])
        rf = set(r[r[f"has_{c}"] == 1]["study_id"]) & missed
        lf = set(l[l[f"has_{c}"] == 1]["study_id"]) & missed
        rows.append({"condition": c, "notes_only_true_cases": len(missed),
                     "both_found": len(rf & lf), "rules_only": len(rf - lf),
                     "llm_only": len(lf - rf), "neither": len(missed - rf - lf)})
    return pd.DataFrame(rows)


def mention_agreement(rules_mentions, llm_mentions) -> dict:
    def keyset(df):
        if df.empty:
            return set()
        return set(zip(df["note_id"], df["concept"], df["status"]))

    r, l = keyset(rules_mentions), keyset(llm_mentions)
    both = len(r & l)
    return {"rules_mentions": len(r), "llm_mentions": len(l), "agreed": both,
            "rules_only": len(r - l), "llm_only": len(l - r),
            "jaccard": round(both / len(r | l), 2) if (r | l) else float("nan")}


def main():
    parser = argparse.ArgumentParser(description="Compare rule-based and AI note reading")
    parser.add_argument("--rules-only", action="store_true", help="skip the AI run (no API key needed)")
    parser.add_argument("--patients", type=int, default=40)
    args = parser.parse_args()

    make_fake_data(n_patients=args.patients)

    print("\n--- Run 1: rules ---")
    rules = run_pipeline(method="rules", verbose=False)
    print(f"  {len(rules['mentions'])} mentions found in notes")

    if args.rules_only:
        print("\n--rules-only: stopping here. Set ANTHROPIC_API_KEY and re-run for the comparison.")
        return

    print("\n--- Smoke test: make_anthropic_caller on one prompt ---")
    print(f"  model replied: {smoke_test_real_caller()}")

    print("\n--- Run 2: LLM ---")
    call_llm, usage, model = make_counting_caller()
    llm = run_pipeline(method="llm", call_llm=call_llm,
                       out_dir="data/output_llm", secure_dir="data/secure_llm", verbose=False)
    print(f"  {len(llm['mentions'])} mentions found in notes")

    print(f"\n=== Accuracy vs answer key (EHR + notes), {model} ===")
    print(accuracy_side_by_side(rules["validation"], llm["validation"]).to_string(index=False))

    print("\n=== Who found the cases the EHR codes missed? ===")
    print(notes_only_agreement(rules["dataset"], llm["dataset"],
                               rules["crosswalk"], "data/raw/truth.csv").to_string(index=False))

    print("\n=== Mention-level agreement (note, concept, status) ===")
    for k, v in mention_agreement(rules["mentions"], llm["mentions"]).items():
        print(f"  {k:<18} {v}")

    rejected = int(llm["audit"]["rule"].str.contains("guardrail", na=False).sum())
    cost = (usage["input_tokens"] / 1e6 * PRICE_IN_PER_MTOK
            + usage["output_tokens"] / 1e6 * PRICE_OUT_PER_MTOK)
    print("\n=== Cost of the LLM run ===")
    print(f"  notes read         {usage['calls']}")
    print(f"  input tokens       {usage['input_tokens']:,}")
    print(f"  output tokens      {usage['output_tokens']:,}")
    print(f"  guardrail rejects  {rejected}")
    print(f"  refusals           {usage['refusals']}")
    print(f"  cost               ${cost:.4f}  (${cost / max(usage['calls'], 1):.5f} per note)")
    print(f"  per 10,000 notes   ${cost / max(usage['calls'], 1) * 10000:.2f}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# KEY POINTS
# - Answers CLAUDE.md next-step 2: does the AI reader beat the rules, and at what cost?
# - The fake data is made ONCE, so both readers see identical notes. Fixed seed, so
#   re-running gives the same answer.
# - Three comparisons, because "better" means different things:
#     * accuracy vs the answer key    - sensitivity and PPV, the epidemiology measures
#     * who found the notes-only cases - the cases EHR codes alone miss, which is the
#       whole reason for reading notes at all
#     * mention-level agreement        - do the two readers mark up the same sentences?
# - The LLM run writes to data/output_llm/ so the rules outputs are not overwritten.
# - Cost is measured from real token counts, not estimated, and projected to 10,000 notes
#   because that is the number that decides whether this is viable at scale.
# - A smoke test calls make_anthropic_caller first: if the request shape is wrong, it
#   fails on one cheap prompt instead of part-way through 40 notes.
# ---------------------------------------------------------------------------
