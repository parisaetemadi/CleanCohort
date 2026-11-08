"""Audit log: a record of every change made to the data."""

import pandas as pd

COLUMNS = ["step", "source", "record", "field", "old_value", "new_value", "rule"]


class AuditLog:
    def __init__(self):
        self.rows = []

    def log(self, step, source, record, field, old, new, rule):
        self.rows.append({
            "step": step,
            "source": source,
            "record": str(record),
            "field": field,
            "old_value": "" if old is None else str(old),
            "new_value": "" if new is None else str(new),
            "rule": rule,
        })

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=COLUMNS)

    def count(self, rule_contains: str) -> int:
        return sum(rule_contains in r["rule"] for r in self.rows)


# ---------------------------------------------------------------------------
# KEY POINTS
# - Regulators (FDA, Health Canada) and journal reviewers ask "how did you get this
#   number?" The audit log answers that for every single change.
# - Each row = which step, which record, old value, new value, and the rule used.
# - This is a big reason you can't just paste data into a chatbot: a chatbot gives an
#   answer, not a traceable record of how it got there.
# - count("converted") = how many changes used a rule containing "converted".
# ---------------------------------------------------------------------------
