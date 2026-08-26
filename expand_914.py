import re
import pandas as pd

SETTING_COLS = ["Rough", "Coarse", "1:1", "Fine", "Very Fine"]
CHANCE_COLS = [f"{s} Chance" for s in SETTING_COLS]
ITEM_COLS = [f"{s} Item" for s in SETTING_COLS]

# Tokens that are item modifiers, not probability chances
MODIFIER_RE = re.compile(r"^\((x\d+|Refueled|Damaged|Recharged|\d+ rounds?)\)$", re.IGNORECASE)
CHANCE_RE = re.compile(r"^\((\d+)%\)$")
# Standalone chance token optionally followed by a modifier: "(50%)" or "(50%) (x2)"
CHANCE_WITH_MOD_RE = re.compile(r"^\((\d+)%\)(?:\s+(\(.+\)))?$")
# Inline chance: "9x19mm (50%) (15 rounds)" or "Destroyed (50%)"
INLINE_CHANCE_RE = re.compile(r"^(.+?)\s+\((\d+)%\)(?:\s+(\(.+\)))?$")


def parse_cell(cell: str, inp: str) -> list[tuple[str, float]]:
    """Parse a cell into list of (item, chance) tuples. Chances sum to 1."""
    tokens = [t.strip() for t in cell.split(",")]
    items = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # Standalone chance token after item: "Item, (50%)" or "Item, (50%) (x2)"
        if i + 1 < len(tokens) and CHANCE_WITH_MOD_RE.match(tokens[i + 1]):
            m = CHANCE_WITH_MOD_RE.match(tokens[i + 1])
            pct = int(m.group(1))
            if m.group(2):
                token = f"{token}, {m.group(2)}"
            i += 2
            items.append((token, pct / 100))
        elif MODIFIER_RE.match(token) and items:
            items[-1] = (f"{items[-1][0]}, {token}", items[-1][1])
            i += 1
        else:
            # Inline chance: "9x19mm (50%) (15 rounds)" or "Destroyed (50%)"
            m = INLINE_CHANCE_RE.match(token)
            if m:
                name = f"{m.group(1)} {m.group(3)}" if m.group(3) else m.group(1)
                items.append((name, int(m.group(2)) / 100))
            else:
                items.append((token, None))
            i += 1

    if len(items) == 1:
        items[0] = (items[0][0], 1.0)

    # Hard-coded edge cases
    result = []
    for item, chance in items:
        if item == "Activates":
            result.append(("Destroyed", chance))
        elif item == "Multiple Items":
            result.append((pd.NA, pd.NA))
        elif inp == "A7" and item == "Flashbang Grenade" and (pd.isna(chance) or chance <= 0):
            result.append(("Flashbang Grenade", 0.5))
        elif inp == "A7" and item == ".44 Revolver" and (pd.isna(chance) or chance <= 0):
            result.append((".44 Revolver", 0.5))
        elif item == "Randomized Attachments":
            result.append((inp, chance))
        else:
            result.append((item, chance))
    return result


SCP_NAMES = {
    "SCP-018":   "Superball",
    "SCP-127":   "Living Gun",
    "SCP-1344":  "Goggles",
    "SCP-1509":  "Blade of Rebirth",
    "SCP-1576":  "Phonograph",
    "SCP-1853":  "Green Serum",
    "SCP-207":   "Cola",
    "SCP-207?":  "Anti-Cola",
    "SCP-2176":  "Ghostlight",
    "SCP-244-A": "Vase",
    "SCP-244-B": "Vase",
    "SCP-268":   "Invisibility Cap",
    "SCP-500":   "The Panacea Pill Bottle",
}

def rename_scp(name: str) -> str:
    return f"{name} ({SCP_NAMES[name]})" if name in SCP_NAMES else name


df = pd.read_csv("914_outputs.csv")
for col in ["Input"] + SETTING_COLS:
    df[col] = df[col].apply(lambda cell: re.sub(
        r'(SCP-\d+\??(?:-[AB])?)',
        lambda m: rename_scp(m.group(1)),
        str(cell)
    ) if pd.notna(cell) else cell)

rows = []
for _, row in df.iterrows():
    # Parse all 5 setting columns
    parsed = {col: parse_cell(str(row[col]), row["Input"]) for col in SETTING_COLS}
    # Number of duplicate rows = max outcomes across all settings
    n = max(len(v) for v in parsed.values())

    for i in range(n):
        new_row = {"Input": row["Input"], "admin_only": row["admin_only"]}
        for col in SETTING_COLS:
            outcomes = parsed[col]
            if i < len(outcomes):
                item, chance = outcomes[i]
            else:
                item, chance = None, None
            new_row[f"{col} Item"] = item
            new_row[f"{col} Chance"] = chance
        rows.append(new_row)

TRAILING_MOD_RE = re.compile(r"^(.+?)[,\s]\s*(\((?:x\d+|Refueled|Damaged|Recharged|\d+ rounds?)\))$", re.IGNORECASE)

out = pd.DataFrame(rows, columns=["Input", "admin_only"] + [c for col in SETTING_COLS for c in (f"{col} Item", f"{col} Chance")])

for col in SETTING_COLS:
    item_col = f"{col} Item"
    mod_col = f"{col} Modifier"
    def split_modifier(val):
        if pd.isna(val):
            return val, None
        m = TRAILING_MOD_RE.match(val)
        return (m.group(1), m.group(2)) if m else (val, None)
    split = out[item_col].map(split_modifier)
    out[item_col] = split.map(lambda x: x[0])
    out.insert(out.columns.get_loc(item_col) + 2, mod_col, split.map(lambda x: x[1]))

out.to_csv("914_outputs_expanded.csv", index=False)
print(f"Saved 914_outputs_expanded.csv ({len(out)} rows)")
