import subprocess
import sys
import pandas as pd

raw = pd.read_csv("914_outputs.csv")[["Input", "category"]].drop_duplicates("Input").reset_index(drop=True)
exp = pd.read_csv("914_outputs_expanded.csv")[["Input", "admin_only"]].drop_duplicates("Input").reset_index(drop=True)
df = exp.copy()
df["category"] = raw["category"]
admin = "--admin" in sys.argv

for _, r in df.iterrows():
    if r["category"] in ("Ammo", "Remote Admin Only Items"):
        continue
    args = [sys.executable, "graph_914.py", r["Input"], "--no-show"]
    print(f"Generating: {r['Input']}", flush=True)
    subprocess.run(args, check=True)
