"""Step 7 - Quality report: an HTML page showing what was fixed, what's missing and what needs review."""

import html

import pandas as pd

CSS = """
body{font-family:system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#1f2933;background:#fff}
h1{font-size:24px} h2{font-size:18px;margin-top:32px;border-bottom:1px solid #e4e7eb;padding-bottom:4px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{border:1px solid #e4e7eb;border-radius:8px;padding:12px}
.card b{display:block;font-size:22px} .card span{font-size:13px;color:#52606d}
table{border-collapse:collapse;width:100%;font-size:13px} td,th{border-bottom:1px solid #e4e7eb;padding:6px;text-align:left}
.bar{background:#e4e7eb;border-radius:4px;height:10px;width:160px;display:inline-block;vertical-align:middle}
.bar i{display:block;height:10px;border-radius:4px;background:#3b82f6}
.warn{color:#b45309}
"""


def _table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "<p>None 🎉</p>"
    return df.head(max_rows).to_html(index=False, escape=True, border=0)


def _card(value, label):
    return f"<div class='card'><b>{html.escape(str(value))}</b><span>{html.escape(label)}</span></div>"


def build_report(stats: dict, dataset: pd.DataFrame, audit_df: pd.DataFrame,
                 unmapped: pd.DataFrame, mentions: pd.DataFrame, unlinked: pd.DataFrame) -> str:
    rule_counts = audit_df.groupby("rule").size().rename("count").reset_index().sort_values("count", ascending=False)

    cols = [c for c in dataset.columns if c != "study_id" and not c.endswith("_source")]
    miss_rows = ""
    for c in cols:
        pct = dataset[c].notna().mean() * 100
        miss_rows += (f"<tr><td>{html.escape(c)}</td><td><span class='bar'><i style='width:{pct:.0f}%'></i>"
                      f"</span> {pct:.0f}%</td></tr>")

    status_counts = (mentions.groupby("status").size().rename("count").reset_index()
                     if not mentions.empty else pd.DataFrame())
    review = audit_df[audit_df["rule"].str.contains("review|conflict|could not link", regex=True)]

    return f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>CleanCohort Quality Report</title><style>{CSS}</style></head><body>
<h1>CleanCohort quality report</h1>
<div class='cards'>
{_card(stats['patients'], 'patients in final dataset')}
{_card(stats['duplicates_removed'], 'duplicate rows removed')}
{_card(stats['unit_conversions'], 'unit conversions')}
{_card(stats['impossible_values'], 'impossible values removed')}
{_card(stats['identifiers_scrubbed'], 'notes de-identified')}
{_card(stats['unmapped_terms'], 'unmapped diagnosis terms')}
{_card(stats['unlinked_records'], 'records not linked')}
{_card(stats['smoking_conflicts'], 'smoking conflicts')}
</div>
<h2>Needs a human to review <span class='warn'>({len(review)})</span></h2>{_table(review[['step','source','record','old_value','new_value','rule']])}
<h2>Unmapped diagnosis terms</h2><p>Add these to <code>vocab.py</code> if they are real synonyms, then re-run.</p>{_table(unmapped)}
<h2>Note reading: mention status</h2><p>Only <b>affirmed</b> mentions count as the patient's condition.</p>{_table(status_counts)}
<h2>Unlinked records</h2>{_table(unlinked)}
<h2>How complete is each variable?</h2><table>{miss_rows}</table>
<h2>All changes by rule (from the audit log)</h2>{_table(rule_counts, 50)}
</body></html>"""


# ---------------------------------------------------------------------------
# KEY POINTS
# - A data tool must SHOW its work. This report is what a researcher or a hospital
#   privacy officer reads before trusting the dataset.
# - "Needs a human to review" collects every fuzzy match, unlinked record and conflict,
#   so a person only checks the uncertain cases, not all records.
# - The completeness bars show missing data per variable at a glance.
# - Every number in the report comes from the audit log, so it is traceable.
# ---------------------------------------------------------------------------
