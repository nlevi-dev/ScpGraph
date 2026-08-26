import re
import pandas as pd

raw = pd.read_csv("914_outputs.csv")[["Input", "category"]].drop_duplicates("Input")
exp = pd.read_csv("914_outputs_expanded.csv")[["Input", "admin_only"]].drop_duplicates("Input")

# raw Input order matches expanded Input order (same rows, just SCP names expanded)
# merge on position within each original item group
raw = raw.reset_index(drop=True)
exp = exp.reset_index(drop=True)
df = exp.copy()
df["category"] = raw["category"]

CATEGORY_ORDER = [
    "Keycards", "Standard Weaponry", "Special Weaponry",
    "Treatment Items", "SCP Items", "Miscellaneous",
]

def slug(name):
    return "graph_full_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ".html"

rows_html = []
for cat in CATEGORY_ORDER:
    items = df[df["category"] == cat]
    if items.empty:
        continue
    rows_html.append(f'<tr><th colspan="2" class="cat">{cat}</th></tr>')
    for _, r in items.iterrows():
        name = r["Input"]
        rows_html.append(f'<tr><td><a href="items/{slug(name)}">{name}</a></td></tr>')

desc = ("<p>This site shows the most efficient crafting paths through <b>SCP-914</b> in "
        "<i>SCP: Secret Laboratory</i> &mdash; the anomalous machine that transforms items "
        "depending on the setting used (Rough, Coarse, 1:1, Fine, Very Fine).</p>"
        "<p>For each target item, the graph displays the <b>optimal, stable route</b>: "
        "starting items are connected by arrows coloured by setting, showing exactly what "
        "to put in and on which setting to maximise your chances. Paths are chosen by "
        "finding the sequence of 914 uses that gives the highest overall probability of "
        "reaching the target, preferring shorter chains when probabilities are equal.</p>"
        "<p>Every non-target node is annotated with three metrics:<br>"
        "&bull; <b>Reach %</b> &mdash; the probability that starting from this item you "
        "eventually produce the target by following the shown path.<br>"
        "&bull; <b>Items</b> &mdash; how many copies of this item you need on average to "
        "get one target (1 &divide; reach).<br>"
        "&bull; <b>Steps</b> &mdash; the expected total number of 914 uses until the "
        "target appears, including failed attempts and all items consumed along the way.</p>")
github_note = ("<p style=\"font-size:.9em;color:#555\">Interested in the technical groundwork? "
               "Check out the source on <a href=\"https://github.com/nlevi-dev/ScpGraph\" target=\"_blank\">GitHub</a>.</p>")
html = ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\">"
        "<title>SCP-914 Crafting Graphs</title>"
        "<style>body{font-family:sans-serif;max-width:700px;margin:2rem auto}"
        "table{border-collapse:collapse;width:100%}td,th{padding:.4rem .8rem;"
        "text-align:left;border:1px solid #ccc}th.cat{background:#2c3e50;color:#fff}"
        "a{text-decoration:none;color:#2980b9}a:hover{text-decoration:underline}"
        "p{color:#333;line-height:1.6}</style></head><body>"
        "<h1>SCP-914 Crafting Graphs</h1>"
        + github_note + desc +
        "<table>" + "".join(rows_html) + "</table></body></html>")

out = "index/index.html"
with open(out, "w") as f:
    f.write(html)
print(f"Saved {out}")
