#!/usr/bin/env python3
import argparse
import json
import tempfile
import shutil
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from drone_solution_benchmark import (
    benchmark_dynamic_drone_solution,
    step1_generate_initial_paths,
    create_shared_environment,
)
from lns2.build import my_lns2 as OriginalMyLns2
from GridBasedPathPlanning.Plan.TJPS import TJPS

def _to_json_serializable(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

def aggregate_metrics(results):
    sums, counts = defaultdict(float), defaultdict(int)
    for res in results:
        for k, v in res.items():
            if isinstance(v, (int, float)):
                sums[k] += v; counts[k] += 1
            elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                sums[k] += float(np.mean(v)); counts[k] += 1
            elif isinstance(v, dict) and all(isinstance(x, (int, float)) for x in v.values()):
                for subk, subv in v.items():
                    sums[f"{k}.{subk}"] += subv
                    counts[f"{k}.{subk}"] += 1
    return {k: sums[k] / counts[k] for k in sums}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map",       default="GridBasedPathPlanning/Data/BenchmarkMaps/Paris_1_256.map")
    p.add_argument("--agents",    type=int, default=5)
    p.add_argument("--obstacles", type=int, default=15)
    p.add_argument("--runs",      type=int, default=10)
    p.add_argument("--output",    default="aggregate_results.json")
    args = p.parse_args()

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    all_runs = {
        "Original LNS2": [],
        "TJPS": [],
        "Dynamic Drone (Before Kinematic)": [],
        "Dynamic Drone Solution": [],
    }
    manh = None

    for run in range(args.runs):
        print(f"\n=== Run {run+1}/{args.runs} ===")

        # 1) Build shared environment
        _, dyn_seq, static_seq, start_list, goal_list = create_shared_environment(
            args.map, args.agents, args.obstacles, frames=1000
        )
        if manh is None:
            manh = [int(abs(s[0]-g[0]) + abs(s[1]-g[1])) for s,g in zip(start_list, goal_list)]

        tmpdir = tempfile.mkdtemp(prefix="bench_")

        # ---- Original LNS2 ----
        t0 = time.time()
        lns2 = OriginalMyLns2.MyLns2(42, static_seq, start_list, goal_list, args.agents, static_seq.shape[1])
        lns2.setUseRerun(False)
        lns2.init_pp()
        it, coll = 0, lns2.calculate_sipps(list(range(args.agents)))
        init_coll = coll
        while coll > 0 and it < 10:
            it += 1
            coll = lns2.calculate_sipps(list(range(args.agents)))
        rt = time.time() - t0

        lengths = [
            sum(abs(p[i+1][0]-p[i][0]) + abs(p[i+1][1]-p[i][1]) for i in range(len(p)-1))
            for p in lns2.vector_path
        ]

        all_runs["Original LNS2"].append({
            "algorithm":        "Original LNS2",
            "runtime":          rt,
            "path_lengths":     lengths,
            "avg_path_length":  float(np.mean(lengths)),
            "success_rate":     1.0 if coll==0 else 1.0-(coll/init_coll),
            "iterations":       it,
            "final_collisions": coll
        })

        # ---- TJPS inline ----
        t0 = time.time()
        tjps_paths, tjps_rts, tjps_succ = [], [], []
        for aid in range(args.agents):
            st = time.time()
            planner = TJPS(
                dyn_seq,
                start_list[aid],
                goal_list[aid],
                obs_threshold=0.1,
                goal_threshold=1,
                max_wait=10,
                max_execution_time=60
            )
            out = planner.plan()

            # unpack path + success
            if isinstance(out, tuple) and len(out) == 2:
                path, success = out

            elif isinstance(out, dict):
                path = out.get("path_extracted", [])
                # handle lists, numpy arrays, etc
                if isinstance(path, np.ndarray):
                    success = path.size > 0
                else:
                    success = hasattr(path, '__len__') and len(path) > 0

            else:
                path = out
                if isinstance(path, np.ndarray):
                    success = path.size > 0
                else:
                    try:
                        success = len(path) > 0
                    except Exception:
                        success = False

            tjps_paths.append(path)
            tjps_succ.append(success)
            tjps_rts.append(time.time() - st)

        rt2 = time.time() - t0
        # only keep the successful paths when computing lengths
        good = [p for p,s in zip(tjps_paths, tjps_succ) if s]
        lengths2 = [
            sum(abs(p[i+1][0]-p[i][0]) + abs(p[i+1][1]-p[i][1]) for i in range(len(p)-1))
            for p in good
        ]

        all_runs["TJPS"].append({
            "algorithm":        "TJPS",
            "runtime":          rt2,
            "agent_runtimes":   tjps_rts,
            "path_lengths":     lengths2,
            "avg_path_length":  float(np.mean(lengths2)) if lengths2 else 0,
            "success_rate":     sum(tjps_succ) / args.agents,
            "iterations":       1,
            "final_collisions": args.agents - sum(tjps_succ)
        })

        # ---- Before Kinematic ----
        paths, _, _, _ = step1_generate_initial_paths(
            args.map, args.agents, args.obstacles, tmpdir
        )
        lens = [
            sum(abs(p[i+1][0]-p[i][0]) + abs(p[i+1][1]-p[i][1]) for i in range(len(p)-1))
            for p in paths
        ]
        all_runs["Dynamic Drone (Before Kinematic)"].append({
            "algorithm":        "Dynamic Drone (Before Kinematic)",
            "runtime":          None,
            "path_lengths":     lens,
            "avg_path_length":  float(np.mean(lens)),
            "success_rate":     1.0,
            "iterations":       0,
            "final_collisions": 0
        })

        # ---- Full Dynamic Drone Solution ----
        r4 = benchmark_dynamic_drone_solution(
            args.map, args.agents, args.obstacles, output_dir=tmpdir
        )
        if r4:
            all_runs["Dynamic Drone Solution"].append(r4)

        shutil.rmtree(tmpdir)

    # ---- Aggregate ----
    aggregated = {alg: aggregate_metrics(rs) for alg,rs in all_runs.items()}

    # ---- Build final JSON ----
    out = {
        "metadata": {
            "manhattan_distances":    manh,
            "avg_manhattan_distance": float(np.mean(manh)),
            "num_agents":             args.agents,
            "timestamp":              timestamp
        },
        "results": []
    }
    for alg, m in aggregated.items():
        entry = {"algorithm": alg}
        entry.update(m)
        out["results"].append(entry)

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=_to_json_serializable)
    print(f"Wrote {args.output}")

    # ---- Plots ----
    df = pd.DataFrame(out["results"])
    def bar(col, title, ylab, fn):
        plt.figure(figsize=(8,5))
        plt.bar(df["algorithm"], df[col])
        plt.title(title)
        plt.ylabel(ylab)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()

    bar("runtime",          "Runtime Comparison",       "Seconds",       "runtime_comparison.png")
    bar("avg_path_length",  "Average Path Length",      "Length",        "path_length_comparison.png")
    bar("success_rate",     "Success Rate Comparison",  "Success rate",  "success_rate_comparison.png")
    print("Saved comparison plots.")

if __name__=="__main__":
    main()