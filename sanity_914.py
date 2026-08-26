import pandas as pd

df = pd.read_csv("914_outputs_expanded.csv")

item_cols = ["Input", "Rough Item", "Coarse Item", "1:1 Item", "Fine Item", "Very Fine Item"]
counts = pd.concat([df[c] for c in item_cols]).dropna().value_counts().rename_axis("Item").reset_index(name="Count").sort_values("Item").reset_index(drop=True)
print(counts.to_string())
