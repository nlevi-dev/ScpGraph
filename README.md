# ScpGraph

Visualises the optimal crafting paths through **SCP-914** (the item-upgrading machine in *SCP: Secret Laboratory*) to reach any target item. Given a target, it builds a directed graph showing which items to feed into 914, on which setting, and annotates every node with the probability of eventually producing the target, the expected number of input items needed, and the expected number of 914 uses.

---

## Pipeline

```
scrape_914.py  →  914_outputs.csv
expand_914.py  →  914_outputs_expanded.csv
graph_914.py   →  graphs/graph_<target>.png  (+ interactive window)
```

### 1. `scrape_914.py`

Fetches the SCP-914 output table from the [SCP:SL wiki](https://en.scpslgame.com/index.php?title=SCP-914/Outputs) and writes `914_outputs.csv`.

Columns: `Input`, `Rough`, `Coarse`, `1:1`, `Fine`, `Very Fine`, `admin_only`.

Each cell contains the human-readable output string exactly as it appears on the wiki (e.g. `"Scientist Keycard, (50%), Research Supervisor Keycard, (50%)"`).

### 2. `expand_914.py`

Parses the compact wiki format into a normalised, one-outcome-per-row CSV (`914_outputs_expanded.csv`).

- Splits probabilistic outcomes into separate rows.
- Extracts inline chances like `9x19mm (50%) (15 rounds)` into separate `Item`, `Chance`, and `Modifier` columns for each setting.
- Expands SCP item names to their full form (e.g. `SCP-207` → `SCP-207 (Cola)`).
- Handles edge cases: `Activates` → `Destroyed`, `Randomized Attachments` → the input item itself, `Multiple Items` → `NA`.
- Single-outcome cells get `Chance = 1.0`.

Output columns (per setting `S` ∈ {Rough, Coarse, 1:1, Fine, Very Fine}):
`Input`, `admin_only`, `S Item`, `S Chance`, `S Modifier`

### 3. `graph_914.py`

The main script. Reads `914_outputs_expanded.csv`, builds a `MultiDiGraph`, prunes it to the most efficient paths toward the target, computes statistics, and renders the graph.

---

## Usage

```bash
# 1. Fetch latest data (only needed once, or when the wiki updates)
python scrape_914.py

# 2. Normalise into expanded CSV
python expand_914.py

# 3. Generate graph for a target item
python graph_914.py "O5 Keycard"
python graph_914.py "MTF-E11-SR" --admin
python graph_914.py "O5 Keycard" --items 20
python graph_914.py "O5 Keycard" --steps 50
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `target` | positional string | The item you want to produce. Case-insensitive; partial/fuzzy matching is supported (Jaro-Winkler ≥ 0.9). |
| `--admin` | flag | Include Remote Admin–only items (e.g. Containment Engineer Keycard, Lantern). Excluded by default. |
| `--items N` | float | Hide any node whose expected input-item cost exceeds N. Those nodes' incoming edges are redirected to **Lost**. |
| `--steps N` | float | Hide any node whose expected 914-use count exceeds N. Same redirect behaviour as `--items`. |

`--items` and `--steps` can be combined; a node is hidden if it exceeds *either* threshold.

---

## How the graph is built

### Graph construction

Every row in the expanded CSV becomes a directed edge `Input → Output` labelled with the setting, chance, count (from modifiers like `(x12)`), and a `group` ID. Edges that share a group ID are **co-produced** — they fire together as one atomic outcome. Currently the only co-produced pair is A7 on Very Fine, which always yields both a `.44 Revolver` and a `Flashbang Grenade` simultaneously (encoded in the CSV as `.44 Revolver` with modifier `(+Flashbang Grenade)`). This generalises correctly to any probability: a 50% chance to get both items would simply be two edges in the same group with `chance=0.5`.

### Pruning to optimal paths

The goal is to show only the *best* route from each item to the target, not every possible path.

**Step 1 — Reachability filter.**
Only nodes that can reach the target (via any path in the full graph) are kept. All outgoing edges from the target itself are removed.

**Step 2 — DP score.**
A fixed-point iteration computes `score[n]` = the best single-path probability of reaching the target from node `n`. Edges are evaluated per group: a group's score is `chance × max(score[dest] for dest in group)`. This converges in at most 1000 iterations.

**Step 3 — Best-group pruning.**
For each node, only the group with the highest score is kept (all edges in that group are retained together). Ties are broken by distance to target (closer wins), then alphabetically on the primary destination.

**Step 4 — Setting tiebreaker.**
If a node still has groups from multiple 914 settings after step 3, only the setting with the fewest groups is kept. Among ties, the highest-index setting (Very Fine > Fine > 1:1 > Coarse > Rough) wins.

**Step 5 — Sibling re-add.**
After pruning to a single best group and setting per node, *all other outcomes of that same setting* are added back. This is important: if Fine on a Scientist Keycard leads to Research Supervisor Keycard (the best path), the graph also shows that Fine can produce Facility Manager Keycard at 33% — because that's a real outcome of the same action. Destinations that exist in the full reachable set are shown as-is; others become **Lost**.

**Step 6 — New-node pruning.**
Nodes introduced by sibling re-add that weren't in the original pruned graph get the same best-group + setting tiebreaker treatment, using reach scores computed on the full reachable subgraph.

**Step 7 — Final cleanup.**
Self-loops are removed. Nodes that no longer have a path to the target are dropped. **Destroyed** and **Lost** are kept only if something points to them.

### Layout optimisation

After all pruning, nodes are arranged in horizontal layers (target at layer 0, deepest ancestors furthest down). **Destroyed** and **Lost** are placed at `max(predecessor_layer) + 1` rather than always at the bottom, so they sit close to the nodes that feed them.

Node order within each layer is then optimised to minimise a weighted crossing cost:

- **Same-layer edges** (hop = 0): cost = number of nodes physically between the two endpoints in that layer.
- **k-hop edges** (k ≥ 1): for each pair of edges where one has hop k and the other has hop ≤ k, add 1/k if they cross geometrically (checked via linear interpolation over the shared layer range, extended ±0.5 layers to catch crossings at shared endpoints).

The optimiser precomputes all candidate crossing pairs once (`_build_edge_pairs`), then evaluates cost cheaply via `_cost_from_pos_x`. Search strategy: sliding window of 2 adjacent layers, stride 1. For layers with ≤ 5 nodes, all permutation pairs are tried exhaustively (max 5! × 5! = 14 400). For larger layers, pairwise swaps are used. Multiple random restarts are run; the globally best result is kept.

### Reach probability

After pruning, a second fixed-point iteration computes `reach[n]` — the probability that starting from item `n` and following the graph's edges, you eventually produce the target. The formula accounts for grouped co-products, multi-output settings (e.g. `(x12)` coins), and independent settings:

```
reach[n] = 1 − ∏_settings (1 − Σ_groups chance × (1 − ∏_members (1 − reach[dest])^count))
```

Within each group, all members fire together — the group contributes the probability that at least one member eventually reaches the target. `reach[target] = 1.0` by definition.

### Expected items (`avg_items`)

```
avg_items[n] = 1 / reach[n]
```

The expected number of copies of item `n` you need to start with to get one target.

### Expected steps (`avg_steps`)

The expected total number of 914 uses from a single starting item `n` until the target is produced, modelled as a FIFO queue (all produced items are also processed). This is solved as a linear system:

```
f[n] = 1 + Σ (chance × count × f[dest])   for live dest
```

then `avg_steps[n] = f[n] / reach[n]`.

If the linear system is degenerate (can happen with `(xN)` cycles), the script falls back to Monte Carlo simulation (10 000 runs per node).

---

## Output

**`graphs/graph_<target>.png`** — saved at 150 dpi into a `graphs/` directory next to the script (created automatically). The filename is derived from the target name and active flags: lowercased, non-alphanumeric characters replaced with underscores, prefixed with `graph_` and optional segments `admin_`, `items_N_`, `steps_N_` — or `full_` if neither `--items` nor `--steps` are set — e.g. `graph_full_o5_keycard.png`, `graph_admin_items_10_steps_50_o5_keycard.png`. An interactive matplotlib window also opens.

### Node colours

| Colour | Meaning |
|---|---|
| Green | Target item |
| Red | Destroyed / Lost |
| Orange | Item that has at least one edge leading to Destroyed or Lost |
| Blue | All other items |

### Node labels

Non-target, non-terminal nodes are labelled:
```
Item Name
reach% | avg_items items | avg_steps steps
```

Example: `Scientist Keycard\n45.2% | 2.2 items | 3.8 steps`

Labels are rendered with a semi-transparent white rounded background box for readability.

### Edge colours (by setting)

| Colour | Setting |
|---|---|
| Red | Rough |
| Orange | Coarse |
| Yellow | 1:1 |
| Green | Fine |
| Blue | Very Fine |

Edges are drawn as gradient lines (dark at source, bright at destination) with arrowheads. Bidirectional edges between the same pair of nodes are offset and always drawn straight. Unidirectional edges whose straight line passes within the radius of an intermediate node are automatically bent with a quadratic Bézier curve. The bend magnitude is found via binary search (starting at `EDGE_BEND`, halving the interval each step, up to 8 iterations) independently for both perpendicular directions; the direction yielding the smaller collision-free magnitude wins. If no candidate clears all nodes within 8 steps, the default `EDGE_BEND` magnitude is used.

---

## Dependencies

```
requests
beautifulsoup4
pandas
networkx
matplotlib
numpy
jellyfish   # optional — enables fuzzy target name matching
```

Install with:
```bash
pip install requests beautifulsoup4 pandas networkx matplotlib numpy jellyfish
```

---

## Files

| File | Description |
|---|---|
| `scrape_914.py` | Scrapes wiki → `914_outputs.csv` |
| `expand_914.py` | Normalises CSV → `914_outputs_expanded.csv` |
| `graph_914.py` | Builds, prunes, and renders the graph |
| `sanity_914.py` | Prints item frequency counts across all columns (debug helper) |
| `914_outputs.csv` | Raw scraped data |
| `914_outputs_expanded.csv` | Normalised, one-outcome-per-row data |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
