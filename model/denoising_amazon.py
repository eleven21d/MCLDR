# model/denoising_amazon.py
import os
import argparse
import networkx as nx
import numpy as np
import json
from collections import defaultdict


def read_edges_threecols(path, rel):
    edges = []
    with open(path, 'r') as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 2:
                continue
            src, dst = int(parts[0]), int(parts[1])
            edges.append((src, dst, rel))
    return edges


def run(data_dir, out_dir, beta=0.4, cold_deg_thresh=2, auto=True):
    os.makedirs(out_dir, exist_ok=True)

    # dataset
    files = os.listdir(data_dir)
    mooc_rels = {"uc", "uk", "cv", "vk", "ct"}
    amazon_rels = {"ui", "ic", "ib"}

    if any(f.startswith("uc") or f.startswith("uk") for f in files):
        dataset_type = "mooc"
        rels = sorted(list(mooc_rels))
        type_map = {
            "uc": ("user", "course"),
            "uk": ("user", "concept"),
            "cv": ("concept", "video"),
            "vk": ("video", "concept"),
            "ct": ("course", "teacher")
        }
        print("MOOC dataset")
        beta_map = defaultdict(lambda: beta)
        min_ui_ratio = None
    elif any(f.startswith("ui") or f.startswith("ic") for f in files):
        dataset_type = "amazon"
        rels = sorted(list(amazon_rels))
        type_map = {
            "ui": ("user", "item"),
            "ic": ("item", "category"),
            "ib": ("item", "brand")
        }
        print("Amazon dataset")
        beta_map = {"ui": 0.02, "ic": 0.10, "ib": 0.10}
        min_ui_ratio = 0.9
        cold_deg_thresh = 1
    else:
        raise RuntimeError("no type dataset")

    candidates = []
    for rel in rels:
        p = os.path.join(data_dir, f"{rel}.txt")
        if os.path.exists(p):
            candidates.append((p, rel))
    if not candidates:
        raise RuntimeError("no files")

    G = nx.Graph()
    merged_edges = []
    for p, rel in candidates:
        raw_edges = read_edges_threecols(p, rel)
        src_type, dst_type = type_map[rel]
        for src, dst, _ in raw_edges:
            u = f"{src_type}_{src}"
            v = f"{dst_type}_{dst}"
            G.add_edge(u, v, etype=rel)
            merged_edges.append((u, v, rel))
    print(f"Node={G.number_of_nodes()}, Edge={G.number_of_edges()}")

    closeness_path = os.path.join(data_dir, "closeness_cache.json")
    if os.path.exists(closeness_path):
        with open(closeness_path, 'r') as f:
            closeness_full = json.load(f)
        closeness_full = {k: float(v) for k, v in closeness_full.items()}
        print(f"Loading,have {len(closeness_full)} nodes")
    else:
        print("computing closeness centrality...")
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc).copy()
            closeness = nx.closeness_centrality(subG)
            closeness_full = {n: closeness.get(n, 0.0) for n in G.nodes()}
        else:
            closeness_full = nx.closeness_centrality(G)
        with open(closeness_path, 'w') as f:
            json.dump(closeness_full, f)
        print(f"files on  {closeness_path}")

    edge_scores = {}
    scores_by_type = defaultdict(list)
    for u, v, rel in merged_edges:
        sc = (closeness_full.get(u, 0.0) + closeness_full.get(v, 0.0)) / 2
        edge_scores[(u, v, rel)] = sc
        scores_by_type[rel].append(sc)

    thresholds = {}
    for rel, vals in scores_by_type.items():
        if not vals:
            continue
        mu, sigma = np.mean(vals), np.std(vals)
        tau = mu + beta_map.get(rel, beta) * sigma
        thresholds[rel] = tau
        print(f"📊 {rel}: mean={mu:.4f}, std={sigma:.4f}, β={beta_map.get(rel, beta):.3f} → τ={tau:.4f}")

    deg = dict(G.degree())
    keep_edges = []
    cold_nodes = set()
    removed_by_type = defaultdict(int)

    for (u, v, rel), sc in edge_scores.items():
        if deg.get(u, 0) <= cold_deg_thresh or deg.get(v, 0) <= cold_deg_thresh:
            keep_edges.append((u, v, rel))
            cold_nodes.update([u, v])
        elif sc >= thresholds[rel]:
            keep_edges.append((u, v, rel))
        else:
            removed_by_type[rel] += 1

    if dataset_type == "amazon":
        ui_total = len(scores_by_type["ui"])
        ui_kept = sum(1 for _, _, r in keep_edges if r == "ui")
        if ui_total > 0 and ui_kept / ui_total < min_ui_ratio:
            print(f"UI save{ui_kept/ui_total:.2f} lower {min_ui_ratio:.2f},make up")
            sorted_ui = sorted(
                [(k, s) for k, s in edge_scores.items() if k[2] == "ui"],
                key=lambda x: x[1], reverse=True
            )
            need = int(min_ui_ratio * ui_total) - ui_kept
            for (u, v, r), _ in sorted_ui[:need]:
                if (u, v, r) not in keep_edges:
                    keep_edges.append((u, v, r))

    out_path = os.path.join(out_dir, f"G_denoised_beta{beta}.txt")
    with open(out_path, 'w') as f:
        for u, v, rel in keep_edges:
            u_id = u.split("_")[1]
            v_id = v.split("_")[1]
            f.write(f"{u_id} {v_id} {rel}\n")

    print("the statistics:")
    for rel in scores_by_type.keys():
        total = len(scores_by_type[rel])
        removed = removed_by_type[rel]
        kept = total - removed
        ratio = kept / total * 100
        print(f"  - {rel}: reserve {kept}/{total} ({ratio:.2f}%)")
    print(f"\ncold-start nodes{len(cold_nodes)}")
    print(f"denoised edge {len(keep_edges)} / {len(merged_edges)}")
    print(f"result in {out_path}")

    return out_path, thresholds, closeness_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--cold_deg_thresh", type=int, default=2)
    args = parser.parse_args()

    run(args.data_dir, args.out_dir, beta=args.beta, cold_deg_thresh=args.cold_deg_thresh)
