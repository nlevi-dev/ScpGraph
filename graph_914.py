#!/usr/bin/env python3
import argparse
import re
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from collections import defaultdict

_ap = argparse.ArgumentParser()
_ap.add_argument("target")
_ap.add_argument("--admin", action="store_true")
_ap.add_argument("--items", type=float, default=None)
_ap.add_argument("--steps", type=float, default=None)
_args = _ap.parse_args()
_raw_target = _args.target
_include_admin = _args.admin
_arg_items = _args.items
_arg_steps = _args.steps

EDGE_SAT_START = 0.3
EDGE_SAT_END = 0.9
EDGE_BEND = 0.2

df = pd.read_csv("914_outputs_expanded.csv")
if not _include_admin:
    df = df[~df["admin_only"]]
SETTINGS = ["Rough", "Coarse", "1:1", "Fine", "Very Fine"]

# Build directed graph: edge = (input -> output)
G = nx.MultiDiGraph()
_group_counter = 0
print("Building graph...", flush=True)
for _, row in df.iterrows():
    src = row["Input"]
    for s in SETTINGS:
        dst = row[f"{s} Item"]
        chance = row[f"{s} Chance"]
        mod = row[f"{s} Modifier"]
        count = 1
        coproduct_item = None
        if pd.notna(mod):
            m = re.match(r'\(x(\d+)\)', str(mod))
            if m:
                count = int(m.group(1))
            cp = re.match(r'^\(\+(.+)\)$', str(mod))
            if cp:
                coproduct_item = cp.group(1)
        if pd.notna(dst) and pd.notna(src):
            gid = _group_counter
            _group_counter += 1
            G.add_edge(src, dst, setting=s, chance=chance, count=count, group=gid, coproduct=False)
            if coproduct_item:
                G.add_edge(src, coproduct_item, setting=s, chance=chance, count=1, group=gid, coproduct=True)

G_full = G.copy()  # save full graph before any pruning

# Resolve target name against known nodes
def _resolve_target(raw, nodes):
    if raw in nodes:
        return raw, None
    lo = raw.lower()
    nodes_lo = {n: n.lower() for n in nodes}
    # lowercase exact
    lc_hits = [n for n, nl in nodes_lo.items() if nl == lo]
    if len(lc_hits) == 1:
        return lc_hits[0], "lowercase"
    # partial unambiguous contains
    part_hits = [n for n, nl in nodes_lo.items() if lo in nl]
    if len(part_hits) == 1:
        return part_hits[0], "partial contains"
    # jaro-winkler >= 0.9
    try:
        from jellyfish import jaro_winkler_similarity as jw
    except ImportError:
        jw = None
    if jw:
        scored = [(jw(lo, nl), n) for n, nl in nodes_lo.items()]
        best_score, best_node = max(scored)
        if best_score >= 0.9:
            return best_node, f"jaro-winkler ({best_score:.3f})"
    return raw, None

target, _match_method = _resolve_target(_raw_target, set(G.nodes()))
if _match_method:
    print(f"Warning: '{_raw_target}' matched '{target}' via {_match_method}", file=sys.stderr)

print(f"Pruning ({len(G.nodes())} nodes, {G.number_of_edges()} edges)...", flush=True)
# Prune: keep only nodes that can reach the target (reverse reachability)
rev = G.reverse()
reachable = nx.descendants(rev, target) | {target}
G = G.subgraph(reachable).copy()

# Remove all outgoing edges from target
G.remove_edges_from(list(G.out_edges(target, keys=True)))

# BFS from target (reversed) to get distance of each node from target
rev = G.reverse()
dist = {target: 0}
frontier = {target}
while frontier:
    next_frontier = set()
    for node in frontier:
        for pred in rev.successors(node):
            if pred not in dist:
                dist[pred] = dist[node] + 1
                next_frontier.add(pred)
    frontier = next_frontier

# DP score for pruning: best single-path chance to target (max over paths)
# Iterate to fixed point to handle cycles correctly.
# For grouped edges (coproducts), score = chance * max(score[dest]) across the group.
def _group_scores(node, g, sc):
    """Return {group_id: (chance, best_dest, best_score)} for non-coproduct primary edges."""
    by_group = defaultdict(list)  # gid -> [(dest, chance, is_coproduct)]
    for _, v, d in g.out_edges(node, data=True):
        by_group[d["group"]].append((v, d.get("chance") or 0, d.get("coproduct", False)))
    result = {}
    for gid, members in by_group.items():
        chance = members[0][1]  # all members share the same chance
        best = max((sc.get(v, 0.0) for v, _, _ in members), default=0.0)
        primary = next((v for v, _, cp in members if not cp), members[0][0])
        result[gid] = (chance, primary, chance * best)
    return result

score = {target: 1.0}
for _ in range(1000):
    new_score = {target: 1.0}
    for node in G.nodes():
        if node == target:
            continue
        gs = _group_scores(node, G, score)
        new_score[node] = max((s for _, _, s in gs.values()), default=0.0)
    if all(abs(new_score.get(n, 0) - score.get(n, 0)) < 1e-12 for n in G.nodes()):
        break
    score = new_score

# Prune edges: for each node keep only the best group (primary dest with max path-chance).
# Coproduct members of the winning group are always kept alongside their primary.
edges_to_remove = []
for node in list(G.nodes()):
    if node == target:
        continue
    gs = _group_scores(node, G, score)
    if not gs:
        continue
    max_score = max(s for _, _, s in gs.values())
    tied = [gid for gid, (_, primary, s) in gs.items() if s == max_score]
    best_gid = min(tied, key=lambda gid: (dist.get(gs[gid][1], float("inf")), gs[gid][1]))
    for u, v, k, d in G.out_edges(node, keys=True, data=True):
        if d["group"] != best_gid:
            edges_to_remove.extend([(node, v, k)])
G.remove_edges_from(edges_to_remove)

# Tiebreaker: if a node still has groups from multiple settings, keep only the
# setting with fewest groups; among ties, keep the highest-index setting.
edges_to_remove = []
for node in list(G.nodes()):
    if node == target:
        continue
    by_setting = defaultdict(set)  # setting -> set of group ids
    for u, v, k, d in G.out_edges(node, keys=True, data=True):
        by_setting[d["setting"]].add(d["group"])
    if len(by_setting) <= 1:
        continue
    best_setting = min(by_setting, key=lambda s: (len(by_setting[s]), -SETTINGS.index(s)))
    for u, v, k, d in G.out_edges(node, keys=True, data=True):
        if d["setting"] != best_setting:
            edges_to_remove.append((u, v, k))
G.remove_edges_from(edges_to_remove)

rev = G.reverse()
G = G.subgraph(nx.descendants(rev, target) | {target}).copy()

print(f"Sibling re-add ({len(G.nodes())} nodes)...", flush=True)
# Re-add sibling edges: for each kept edge (node->best_dest, setting=s),
# add back all edges from node with the same setting from G_full.
# Use G_full reachability for the Lost check so intermediate nodes like
# High-Explosive Grenade aren't incorrectly collapsed to Lost.
_full_reachable = nx.descendants(G_full.reverse(), target) | {target}
_pruned_nodes = set(G.nodes())  # snapshot before sibling re-add
for node in list(G.nodes()):
    if node == target:
        continue
    kept_settings = {d["setting"] for _, _, d in G.out_edges(node, data=True)}
    existing = {(v, d["setting"]) for _, v, d in G.out_edges(node, data=True)}
    for _, v, d in G_full.out_edges(node, data=True):
        if d["setting"] not in kept_settings:
            continue
        dst = v if (v in _full_reachable or v == "Destroyed") else "Lost"
        if (dst, d["setting"]) not in existing:
            G.add_edge(node, dst, **d)
            existing.add((dst, d["setting"]))

# For nodes newly referenced by sibling re-add that weren't in the pruned graph,
# apply the same pruning (best dest by score, then highest-index setting),
# then re-add siblings for the selected setting.
for node in _full_reachable - _pruned_nodes:
    for _, v, d in G_full.out_edges(node, data=True):
        dst = v if (v in _full_reachable or v == "Destroyed") else "Lost"
        G.add_edge(node, dst, **d)

# Remove self-loops
G.remove_edges_from([(u, v, k) for u, v, k in G.edges(keys=True) if u == v])

# Compute reach on full reachable subgraph for accurate new-node pruning
_G_full_sub = G_full.subgraph(_full_reachable).copy()
_G_full_sub.remove_edges_from(list(_G_full_sub.out_edges(target, keys=True)))
_G_full_sub.remove_edges_from([(u, v, k) for u, v, k in _G_full_sub.edges(keys=True) if u == v])
_full_nodes = list(_G_full_sub.nodes())
_full_reach = {n: 0.0 for n in _full_nodes}
_full_reach[target] = 1.0
for _ in range(1000):
    _new_r = {target: 1.0}
    for _n in _full_nodes:
        if _n == target:
            continue
        _by_s_g = defaultdict(lambda: defaultdict(list))  # setting -> group -> [(r, cnt)]
        for _, _v, _d in _G_full_sub.out_edges(_n, data=True):
            _by_s_g[_d["setting"]][_d["group"]].append((_full_reach.get(_v, 0.0), _d.get("chance") or 0, _d.get("count") or 1))
        _new_r[_n] = 1.0 - np.prod([
            1.0 - sum(
                members[0][1] * (1.0 - np.prod([(1.0 - r) ** cnt for r, _, cnt in members]))
                for members in _by_g.values()
            )
            for _by_g in _by_s_g.values()
        ]) if _by_s_g else 0.0
    if max(abs(_new_r[_n] - _full_reach[_n]) for _n in _full_nodes) < 1e-9:
        break
    _full_reach = _new_r

# Apply full pruning to all nodes not yet pruned (newly seeded nodes)
# Use current-graph reachability to avoid picking destinations that can't reach target
_can_reach_now = nx.descendants(G.reverse(), target) | {target}
# Recompute dist on current graph for accurate tiebreaking
_dist_now = {target: 0}
_frontier = {target}
_rev_now = G.reverse()
while _frontier:
    _next = set()
    for _n in _frontier:
        for _p in _rev_now.successors(_n):
            if _p not in _dist_now:
                _dist_now[_p] = _dist_now[_n] + 1
                _next.add(_p)
    _frontier = _next
# Process in order of increasing distance so destinations are pruned before their predecessors
_new_node_order = sorted(
    [n for n in G.nodes() if n != target and n not in _pruned_nodes],
    key=lambda n: _dist_now.get(n, float("inf"))
)
_valid_dests = set(_pruned_nodes) | {target, "Destroyed", "Lost"}
for node in _new_node_order:
    gs = _group_scores(node, G, _full_reach)
    gs = {gid: v for gid, v in gs.items() if v[1] in _valid_dests}
    if not gs:
        continue
    max_s = max(s for _, _, s in gs.values())
    tied = [gid for gid, (_, primary, s) in gs.items() if s == max_s]
    best_gid = min(tied, key=lambda gid: (_dist_now.get(gs[gid][1], float("inf")), gs[gid][1]))
    edges_to_remove = [(u, v, k) for u, v, k, d in G.out_edges(node, keys=True, data=True)
                       if d["group"] != best_gid]
    G.remove_edges_from(edges_to_remove)
    by_setting = defaultdict(set)
    for u, v, k, d in G.out_edges(node, keys=True, data=True):
        by_setting[d["setting"]].add(d["group"])
    if len(by_setting) > 1:
        best_setting = min(by_setting, key=lambda s: (len(by_setting[s]), -SETTINGS.index(s)))
        G.remove_edges_from([(u, v, k) for u, v, k, d in G.out_edges(node, keys=True, data=True)
                              if d["setting"] != best_setting])
    _valid_dests.add(node)

# Also apply same tiebreaker to pruned nodes
for node in _pruned_nodes:
    if node == target:
        continue
    by_setting = defaultdict(set)
    for u, v, k, d in G.out_edges(node, keys=True, data=True):
        by_setting[d["setting"]].add(d["group"])
    if len(by_setting) <= 1:
        continue
    best_setting = min(by_setting, key=lambda s: (len(by_setting[s]), -SETTINGS.index(s)))
    G.remove_edges_from([(u, v, k) for u, v, k, d in G.out_edges(node, keys=True, data=True)
                         if d["setting"] != best_setting])

# Re-add siblings for all nodes using their now-selected setting
for node in list(G.nodes()):
    if node == target:
        continue
    kept_settings = {d["setting"] for _, _, d in G.out_edges(node, data=True)}
    existing = {(v, d["setting"]) for _, v, d in G.out_edges(node, data=True)}
    for _, v, d in G_full.out_edges(node, data=True):
        if d["setting"] not in kept_settings:
            continue
        dst = v if (v in _full_reachable or v == "Destroyed") else "Lost"
        if (dst, d["setting"]) not in existing:
            G.add_edge(node, dst, **d)
            existing.add((dst, d["setting"]))

# Seed and prune any nodes newly referenced by the final sibling re-add
for node in list(G.nodes()):
    if node == target or node in _pruned_nodes:
        continue
    if G.out_degree(node) > 0:
        continue  # already pruned
    for _, v, d in G_full.out_edges(node, data=True):
        dst = v if (v in _full_reachable or v == "Destroyed") else "Lost"
        G.add_edge(node, dst, **d)
    G.remove_edges_from([(u, v, k) for u, v, k in G.out_edges(node, keys=True) if u == v])
    gs = _group_scores(node, G, _full_reach)
    gs = {gid: v for gid, v in gs.items() if v[1] in _valid_dests}
    if not gs:
        continue
    max_s = max(s for _, _, s in gs.values())
    tied = [gid for gid, (_, primary, s) in gs.items() if s == max_s]
    best_gid = min(tied, key=lambda gid: (_dist_now.get(gs[gid][1], float("inf")), gs[gid][1]))
    G.remove_edges_from([(u, v, k) for u, v, k, d in G.out_edges(node, keys=True, data=True)
                         if d["group"] != best_gid])
    by_setting = defaultdict(set)
    for u, v, k, d in G.out_edges(node, keys=True, data=True):
        by_setting[d["setting"]].add(d["group"])
    if len(by_setting) > 1:
        best_setting = min(by_setting, key=lambda s: (len(by_setting[s]), -SETTINGS.index(s)))
        G.remove_edges_from([(u, v, k) for u, v, k, d in G.out_edges(node, keys=True, data=True)
                              if d["setting"] != best_setting])

rev = G.reverse()
keep = nx.descendants(rev, target) | {target}
for special in ("Destroyed", "Lost"):
    if any(v == special for _, v in G.out_edges(keep)):
        keep.add(special)
G = G.subgraph(keep).copy()

# BFS layers from target using reversed edges (ascendants by distance)
rev = G.reverse()
layers = defaultdict(set)
layers_depth = {}  # node -> layer index (excludes Destroyed/Lost until assigned below)
layers[0].add(target)
layers_depth[target] = 0
visited = {target}
frontier = {target}
depth = 0
_specials = {"Destroyed", "Lost"}
while frontier:
    depth += 1
    next_frontier = set()
    for node in frontier:
        for pred in rev.successors(node):
            if pred not in visited and pred not in _specials:
                visited.add(pred)
                layers[depth].add(pred)
                layers_depth[pred] = depth
                next_frontier.add(pred)
    frontier = next_frontier

# Assign Destroyed/Lost to their own layers based on max predecessor depth + 1
special_nodes = [n for n in ("Destroyed", "Lost") if n in G.nodes()]
for sp in special_nodes:
    preds_in_layers = [layers_depth[u] for u, _ in G.in_edges(sp) if u in layers_depth]
    sp_depth = (max(preds_in_layers) + 1) if preds_in_layers else (max(layers.keys()) + 1)
    layers[sp_depth].add(sp)
    layers_depth[sp] = sp_depth

max_depth = max(layers.keys()) if layers else 0

def _make_pos(layer_order):
    p = {}
    for d, ordered in layer_order.items():
        for i, node in enumerate(ordered):
            x = (i - (len(ordered) - 1) / 2) * 2.5
            p[node] = (x, -d * 2)
    return p

# Initial layer ordering: sorted alphabetically
layer_order = {d: sorted(nodes) for d, nodes in layers.items()}
pos = _make_pos(layer_order)

# ---------------------------------------------------------------------------
# Layer-order optimisation: minimise weighted crossing cost
# ---------------------------------------------------------------------------
from itertools import permutations as _perms

def _build_edge_pairs(G, layers_depth):
    """
    Precompute static per-edge data and all pairs that could cross.
    Returns (node_layer, edge_nodes, edge_pairs) where:
      node_layer[node] = (layer, is_lo)  -- which endpoint is the lower-layer one
      edge_nodes[i] = (u, v) normalised so layers_depth[u] <= layers_depth[v]
      edge_pairs = list of (i, j, hop, overlap_lo, overlap_hi, w) for pairs that can cross
    """
    # Deduplicated edge list, normalised lo->hi
    seen = set()
    edges = []  # (u, v, la, lb, hop)  la <= lb
    for u, v, _ in G.edges(keys=True):
        if u not in layers_depth or v not in layers_depth:
            continue
        key = (u, v)
        if key in seen:
            continue
        seen.add(key)
        la, lb = layers_depth[u], layers_depth[v]
        if la > lb:
            u, v, la, lb = v, u, lb, la
        edges.append((u, v, la, lb, abs(lb - la)))

    pairs = []
    n = len(edges)
    for i in range(n):
        u1, v1, la1, lb1, hop1 = edges[i]
        for j in range(i + 1, n):
            u2, v2, la2, lb2, hop2 = edges[j]
            hop = max(hop1, hop2)
            if hop == 0:
                continue  # same-layer pairs handled separately
            if hop2 > hop1:
                continue  # only count pair under the larger-hop edge
            raw_lo = max(la1, la2)
            raw_hi = min(lb1, lb2)
            if raw_lo > raw_hi:
                continue
            union_lo = min(la1, la2)
            union_hi = max(lb1, lb2)
            overlap_lo = max(raw_lo - 0.5, union_lo)
            overlap_hi = min(raw_hi + 0.5, union_hi)
            if overlap_lo >= overlap_hi:
                continue
            pairs.append((i, j, hop1, la1, lb1, la2, lb2, overlap_lo, overlap_hi))

    return edges, pairs


def _cost_from_pos_x(pos_x, edges, pairs, layers):
    """Compute crossing cost given pos_x dict. edges and pairs from _build_edge_pairs."""
    total = 0.0
    # Same-layer edges
    for u, v, la, lb, hop in edges:
        if hop == 0:
            xa, xb = pos_x.get(u, 0), pos_x.get(v, 0)
            total += max(0, abs(xa - xb) - 1)
    # Crossing pairs
    for i, j, hop, la1, lb1, la2, lb2, overlap_lo, overlap_hi in pairs:
        u1, v1 = edges[i][0], edges[i][1]
        u2, v2 = edges[j][0], edges[j][1]
        xa1, xb1 = pos_x.get(u1, 0), pos_x.get(v1, 0)
        xa2, xb2 = pos_x.get(u2, 0), pos_x.get(v2, 0)
        def _x_at(exa, exb, ela, elb, d):
            if ela == elb:
                return (exa + exb) / 2
            return exa + (exb - exa) * (d - ela) / (elb - ela)
        x1a = _x_at(xa1, xb1, la1, lb1, overlap_lo)
        x1b = _x_at(xa1, xb1, la1, lb1, overlap_hi)
        x2a = _x_at(xa2, xb2, la2, lb2, overlap_lo)
        x2b = _x_at(xa2, xb2, la2, lb2, overlap_hi)
        if (x1a - x2a) * (x1b - x2b) < 0:
            total += 1.0 / hop
    return total


def _crossing_cost(layer_order, G, layers_depth):
    edges, pairs = _build_edge_pairs(G, layers_depth)
    pos_x = {n: i for d, ordered in layer_order.items() for i, n in enumerate(ordered)}
    return _cost_from_pos_x(pos_x, edges, pairs, layer_order)


def _optimise_layout(layer_order, G, layers_depth, n_restarts=10, max_passes=10):
    sorted_depths = sorted(layer_order.keys())
    n_layers = len(sorted_depths)
    BRUTE_THRESH = 5
    edges, pairs = _build_edge_pairs(G, layers_depth)

    def _cost(lo):
        pos_x = {n: i for d, ordered in lo.items() for i, n in enumerate(ordered)}
        return _cost_from_pos_x(pos_x, edges, pairs, lo)

    def _one_run(lo):
        lo = {d: list(v) for d, v in lo.items()}
        best_cost = _cost(lo)
        for _pass in range(max_passes):
            prev_cost = best_cost
            for wi in range(n_layers - 1):
                da, db = sorted_depths[wi], sorted_depths[wi + 1]
                if len(lo[da]) <= BRUTE_THRESH and len(lo[db]) <= BRUTE_THRESH:
                    best_a, best_b = lo[da][:], lo[db][:]
                    for pa in _perms(lo[da]):
                        for pb in _perms(lo[db]):
                            lo[da], lo[db] = list(pa), list(pb)
                            c = _cost(lo)
                            if c < best_cost:
                                best_cost = c
                                best_a, best_b = list(pa), list(pb)
                    lo[da], lo[db] = best_a, best_b
                else:
                    improved = True
                    while improved:
                        improved = False
                        for layer in (da, db):
                            for ii in range(len(lo[layer])):
                                for jj in range(ii + 1, len(lo[layer])):
                                    lo[layer][ii], lo[layer][jj] = lo[layer][jj], lo[layer][ii]
                                    c = _cost(lo)
                                    if c < best_cost:
                                        best_cost = c
                                        improved = True
                                    else:
                                        lo[layer][ii], lo[layer][jj] = lo[layer][jj], lo[layer][ii]
            if best_cost >= prev_cost:
                break
        return lo, best_cost

    import random as _rand
    best_lo = {d: list(v) for d, v in layer_order.items()}
    best_cost = _cost(best_lo)
    print(f"Layout optimisation: initial cost={best_cost:.3f}, {n_restarts} restarts...", flush=True)
    for restart in range(n_restarts):
        lo_init = {d: list(v) for d, v in layer_order.items()}
        for d in lo_init:
            _rand.shuffle(lo_init[d])
        lo, cost = _one_run(lo_init)
        print(f"  restart {restart+1}/{n_restarts}: cost={cost:.3f}", flush=True)
        if cost < best_cost:
            best_cost = cost
            best_lo = lo
    print(f"Layout optimisation done: best cost={best_cost:.3f}", flush=True)
    return best_lo

print(f"Computing reach ({len(G.nodes())} nodes, {G.number_of_edges()} edges)...", flush=True)
# Iterative fixed-point: p[n] = sum(chance*(1-(1-p[dest])^count)), p[target]=1
# Converges because all probabilities are in [0,1] and the map is a contraction
_nodes = list(G.nodes())
reach = {n: 0.0 for n in _nodes}
reach[target] = 1.0
for _ in range(1000):
    new = {target: 1.0}
    for node in _nodes:
        if node == target:
            continue
        by_s_g = defaultdict(lambda: defaultdict(list))  # setting -> group -> [(r, chance, cnt)]
        for _, v, d in G.out_edges(node, data=True):
            by_s_g[d["setting"]][d["group"]].append((reach.get(v, 0.0), d.get("chance") or 0, d.get("count") or 1))
        new[node] = 1.0 - np.prod([
            1.0 - sum(
                members[0][1] * (1.0 - np.prod([(1.0 - r) ** cnt for r, _, cnt in members]))
                for members in by_g.values()
            )
            for by_g in by_s_g.values()
        ])
    if max(abs(new[n] - reach[n]) for n in _nodes) < 1e-9:
        break
    reach = new

# avg_items[n] = expected items of type n needed to get one target = 1/reach[n]
avg_items = {n: (1.0 / reach[n] if reach.get(n, 0) > 0 else float("inf")) for n in _nodes}

print("Computing steps...", flush=True)
# avg_edges[n] = expected total 914 uses until target (FIFO queue model, stop on first target)
# f[n] = expected steps per attempt; e[n] = f[n] / reach[n]
# f[n] = 1 + sum(chance * count * f[v]) for live v -- solve via linear system
_dead = {"Destroyed", "Lost"}
_live = [n for n in _nodes if n not in _dead and n != target]
_eidx = {n: i for i, n in enumerate(_live)}
_M = len(_live)
_A = np.eye(_M)
_b = np.ones(_M)
for node in _live:
    i = _eidx[node]
    for _, v, d in G.out_edges(node, data=True):
        c = (d.get("chance") or 0) * (d.get("count") or 1)
        if v in _eidx:
            _A[i, _eidx[v]] -= c
        elif v == "Lost":
            _b[i] += c  # Lost items still cost 1 step each when processed
        # Note: coproduct edges contribute their own chance*count terms correctly
        # because they share the same chance as their primary but are separate edges
print(f"Computing steps ({len(_live)} live nodes)...", flush=True)
_x, _, rank, _ = np.linalg.lstsq(_A, _b, rcond=None)
_f = {n: float(_x[_eidx[n]]) for n in _live}

# If any f values are invalid (negative/zero from singular system due to (xN) cycles),
# fall back to Monte Carlo simulation for all nodes.
if any(v <= 0 for v in _f.values()):
    import random as _rng
    _rng.seed(0)
    # Build group-aware edge structure: node -> [(group_id, chance, [(dest, count)])]
    _groups_by_node = defaultdict(dict)  # node -> {gid: (chance, [(dest, count)])}
    for _u, _v, _d in G.edges(data=True):
        gid = _d["group"]
        if gid not in _groups_by_node[_u]:
            _groups_by_node[_u][gid] = (_d.get("chance") or 0, [])
        _groups_by_node[_u][gid][1].append((_v, _d.get("count") or 1))
    _groups_list = {node: list(gdata.values()) for node, gdata in _groups_by_node.items()}

    def _simulate(start, n=10000):
        total_steps = 0
        successes = 0
        for _ in range(n):
            queue = [start]
            while queue:
                item = queue.pop(0)
                if item == target:
                    successes += 1
                    break
                if item in _dead or item not in _groups_list:
                    break
                r = _rng.random()
                cum = 0.0
                for chance, members in _groups_list[item]:
                    cum += chance
                    if r < cum:
                        total_steps += 1
                        for _v, count in members:
                            for _ in range(count):
                                queue.append(_v)
                        break
        return (total_steps / successes) if successes > 0 else float("inf")

    avg_edges = {}
    for i, n in enumerate(_live):
        print(f"  Simulating {i+1}/{len(_live)}: {n}...", flush=True)
        avg_edges[n] = round(_simulate(n), 1)
else:
    avg_edges = {n: round(_f[n] / reach[n], 1) if reach.get(n, 0) > 0 else float("inf") for n in _live}
avg_edges[target] = 0.0

# Prune nodes exceeding --items / --steps thresholds; redirect their incoming edges to Lost
if _arg_items is not None or _arg_steps is not None:
    _prune = {n for n in G.nodes() if n not in (target, "Destroyed", "Lost")
              and ((_arg_items is not None and avg_items.get(n, float("inf")) > _arg_items)
                   or (_arg_steps is not None and avg_edges.get(n, float("inf")) > _arg_steps))}
    if _prune:
        for u, v, k, d in list(G.in_edges(_prune, keys=True, data=True)):
            if u not in _prune:
                G.add_edge(u, "Lost", **d)
        G.remove_nodes_from(_prune)
        # Drop special nodes if nothing points to them anymore
        for _s in ("Destroyed", "Lost"):
            if _s in G.nodes() and G.in_degree(_s) == 0:
                G.remove_node(_s)
        # Recompute layers/pos after pruning
        rev = G.reverse()
        layers = defaultdict(set)
        layers_depth = {}
        layers[0].add(target)
        layers_depth[target] = 0
        visited = {target}
        frontier = {target}
        depth = 0
        while frontier:
            depth += 1
            next_frontier = set()
            for node in frontier:
                for pred in rev.successors(node):
                    if pred not in visited and pred not in _specials:
                        visited.add(pred)
                        layers[depth].add(pred)
                        layers_depth[pred] = depth
                        next_frontier.add(pred)
            frontier = next_frontier
        special_nodes = [n for n in ("Destroyed", "Lost") if n in G.nodes()]
        for sp in special_nodes:
            preds_in_layers = [layers_depth[u] for u, _ in G.in_edges(sp) if u in layers_depth]
            sp_depth = (max(preds_in_layers) + 1) if preds_in_layers else (max(layers.keys()) + 1)
            layers[sp_depth].add(sp)
            layers_depth[sp] = sp_depth
        max_depth = max(layers.keys()) if layers else 0
        layer_order = {d: sorted(nodes) for d, nodes in layers.items()}
        at_risk = {u for u, v in G.edges() if v in ("Destroyed", "Lost")}
        node_colors = []
        for n in G.nodes():
            if n == target:
                node_colors.append("#2ecc71")
            elif n in ("Destroyed", "Lost"):
                node_colors.append("#e74c3c")
            elif n in at_risk:
                node_colors.append("#e67e22")
            else:
                node_colors.append("#3498db")
        edge_list = list(G.edges(data=True))
        pair_count = defaultdict(int)
        for u, v, _ in edge_list:
            pair_count[tuple(sorted((u, v)))] += 1
        all_pos_list = list(pos.values())

layer_order = _optimise_layout(layer_order, G, layers_depth)
pos = _make_pos(layer_order)

print("Drawing...", flush=True)
# Draw
fig, ax = plt.subplots(figsize=(max(14, len(G.nodes) * 0.6), max(8, len(layers) * 2.5)))

setting_colors = {
    "Rough": "#e8503a", "Coarse": "#e8943a", "1:1": "#d4bc2a",
    "Fine": "#3abf6a", "Very Fine": "#3a7fd4"
}

at_risk = {u for u, v in G.edges() if v in ("Destroyed", "Lost")}
node_colors = []
for n in G.nodes():
    if n == target:
        node_colors.append("#2ecc71")
    elif n in ("Destroyed", "Lost"):
        node_colors.append("#e74c3c")
    elif n in at_risk:
        node_colors.append("#e67e22")
    else:
        node_colors.append("#3498db")

nc = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
nc.set_zorder(2)
ax.autoscale_view()

def draw_gradient_edge(ax, p1, p2, color, ctrl=None, n_segments=30):
    base = np.array(mcolors.to_rgb(color))
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length == 0:
        return
    # Convert node radius from pts to data coords
    r_pts = np.sqrt(800 / np.pi)
    r_inch = r_pts / 72
    ax_w_inch = ax.get_position().width * fig.get_figwidth()
    ax_h_inch = ax.get_position().height * fig.get_figheight()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    rx = r_inch / ax_w_inch * (x1 - x0)
    ry = r_inch / ax_h_inch * (y1 - y0)
    unit = direction / length
    node_r = np.sqrt((unit[0] * rx) ** 2 + (unit[1] * ry) ** 2)
    p1c = p1 + unit * node_r
    p2c = p2 - unit * node_r
    if ctrl is not None:
        # Quadratic bezier: sample points along curve
        ts = np.linspace(0, 1, n_segments + 1)
        pts = np.outer((1-ts)**2, p1c) + np.outer(2*(1-ts)*ts, ctrl) + np.outer(ts**2, p2c)
        xs, ys = pts[:, 0], pts[:, 1]
    else:
        xs = np.linspace(p1c[0], p2c[0], n_segments + 1)
        ys = np.linspace(p1c[1], p2c[1], n_segments + 1)
    for i in range(n_segments):
        t = EDGE_SAT_START + (i / n_segments) * (EDGE_SAT_END - EDGE_SAT_START)
        c = base * t
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color=c, linewidth=1.5, solid_capstyle="butt", zorder=1)
    ax.annotate("", xy=p2c, xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle="->", color=tuple(base * 0.8), lw=1.5), zorder=1)

def needs_bend(p1, p2, all_pos, fig, ax):
    """Return True if any other node's circle (radius in data coords) intersects the line p1->p2."""
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length == 0:
        return False
    unit = direction / length
    r_pts = np.sqrt(800 / np.pi)
    r_inch = r_pts / 72
    ax_w_inch = ax.get_position().width * fig.get_figwidth()
    ax_h_inch = ax.get_position().height * fig.get_figheight()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    rx = r_inch / ax_w_inch * (x1 - x0)
    ry = r_inch / ax_h_inch * (y1 - y0)
    for q in all_pos:
        q = np.array(q)
        if np.allclose(q, p1) or np.allclose(q, p2):
            continue
        t = np.dot(q - p1, unit)
        if t <= 0 or t >= length:
            continue
        closest = p1 + t * unit
        diff = q - closest
        # Elliptical node: check if diff is within the node ellipse
        if (diff[0] / rx) ** 2 + (diff[1] / ry) ** 2 < 1.0:
            return True
    return False

def bend_ctrl(p1, p2, all_pos):
    """Compute a quadratic bezier control point that bends away from intermediate nodes."""
    mid = (p1 + p2) / 2
    direction = p2 - p1
    # Perpendicular (bend to the right of travel direction)
    perp = np.array([-direction[1], direction[0]])
    norm = np.linalg.norm(perp)
    if norm == 0:
        return mid
    perp = perp / norm
    # Bend amount: proportional to edge length
    bend = np.linalg.norm(direction) * EDGE_BEND
    return mid + perp * bend

edge_list = list(G.edges(data=True))
pair_count = defaultdict(int)
for u, v, _ in edge_list:
    pair_count[tuple(sorted((u, v)))] += 1

all_pos_list = list(pos.values())
_edge_artists = []

def redraw_edges(event=None):
    for artist in _edge_artists:
        artist.remove()
    _edge_artists.clear()
    pair_index = defaultdict(int)
    _orig_plot = ax.plot
    _orig_annotate = ax.annotate
    def _plot(*args, **kwargs):
        lines = _orig_plot(*args, **kwargs)
        _edge_artists.extend(lines)
        return lines
    def _annotate(*args, **kwargs):
        ann = _orig_annotate(*args, **kwargs)
        _edge_artists.append(ann)
        return ann
    ax.plot = _plot
    ax.annotate = _annotate
    for u, v, data in edge_list:
        if u not in pos or v not in pos:
            continue
        color = setting_colors.get(data["setting"], "#999")
        p1, p2 = np.array(pos[u]), np.array(pos[v])
        canon = tuple(sorted((u, v)))
        n = pair_count[canon]
        idx = pair_index[canon]
        pair_index[canon] += 1
        if n > 1:
            cp1, cp2 = np.array(pos[canon[0]]), np.array(pos[canon[1]])
            perp = np.array([-(cp2[1]-cp1[1]), cp2[0]-cp1[0]])
            norm = np.linalg.norm(perp)
            if norm > 0:
                perp = perp / norm * 0.2 * (idx - (n-1)/2)
                p1, p2 = p1 + perp, p2 + perp
        bidir = pair_count[canon] > 1
        ctrl = bend_ctrl(p1, p2, all_pos_list) if (not bidir and needs_bend(p1, p2, all_pos_list, fig, ax)) else None
        draw_gradient_edge(ax, p1, p2, color, ctrl=ctrl)
    ax.plot = _orig_plot
    ax.annotate = _orig_annotate

redraw_edges()
fig.canvas.mpl_connect('resize_event', redraw_edges)

nc = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
nc.set_zorder(2)
def _fmt_node(n):
    if n in ("Destroyed", "Lost") or n == target:
        return n
    p = reach.get(n, 0)
    ai = avg_items.get(n, float("inf"))
    ae = avg_edges.get(n, float("inf"))
    ai_str = f"{ai:.1f}" if ai != float("inf") else "∞"
    ae_str = f"{ae:.1f}" if ae != float("inf") else "∞"
    return f"{n}\n{p*100:.1f}% | {ai_str} items | {ae_str} steps"
node_labels = {n: _fmt_node(n) for n in G.nodes()}
for t in nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7, ax=ax).values():
    t.set_zorder(3)

legend = [mpatches.Patch(color=c, label=s) for s, c in setting_colors.items()]
ax.legend(handles=legend, loc="upper right")
ax.set_title(f"Paths to: {target}", y=0.98)
ax.axis("off")
plt.tight_layout()
import os
_graphs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
os.makedirs(_graphs_dir, exist_ok=True)
_fname = "graph_"
if _include_admin:
    _fname += "admin_"
if _arg_items is not None:
    _fname += f"items_{int(_arg_items)}_"
if _arg_steps is not None:
    _fname += f"steps_{int(_arg_steps)}_"
if _arg_items is None and _arg_steps is None:
    _fname += "full_"
_fname += re.sub(r'[^a-z0-9]+', '_', target.lower()).strip('_') + ".png"
_fpath = os.path.join(_graphs_dir, _fname)
plt.savefig(_fpath, dpi=150)
print(f"Saved {_fpath}")
plt.show()
