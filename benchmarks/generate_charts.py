#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate publication-quality benchmark charts for DeepSeek-V4.1-Flash on Multi-Host TPU v6e."""

import argparse
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def load_all_results(results_dir: Path) -> list[dict]:
    records = []
    for p in sorted(results_dir.glob("bench_*.json")):
        data = json.loads(p.read_text())
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict) and "cells" in data:
            records.extend(data["cells"])
    return records


def save_summary_csv(records: list[dict], out_csv: Path) -> None:
    if not records:
        return
    fieldnames = [
        "scenario",
        "target_input_tokens",
        "target_output_tokens",
        "concurrency",
        "num_requests",
        "successful_requests",
        "wall_time_s",
        "prefill_tok_per_s",
        "decode_tok_per_s",
        "total_tok_per_s",
        "requests_per_s",
        "ttft_mean_ms",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "ttft_p99_ms",
        "tpot_mean_ms",
        "tpot_p50_ms",
        "tpot_p90_ms",
        "e2e_mean_ms",
        "e2e_p50_ms",
        "e2e_p90_ms",
        "e2e_p99_ms",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"Saved summary CSV ({len(records)} rows) -> {out_csv}")


def plot_throughput_vs_concurrency(records: list[dict], charts_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    scenarios = sorted(set(r["scenario"] for r in records if "opt" in r["scenario"] or "agentic" in r["scenario"]))
    if not scenarios:
        scenarios = sorted(set(r["scenario"] for r in records))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "P"]

    for idx, scen in enumerate(scenarios):
        sub = sorted([r for r in records if r["scenario"] == scen], key=lambda x: x["concurrency"])
        if not sub:
            continue
        c = [r["concurrency"] for r in sub]
        tot_tps = [r["total_tok_per_s"] for r in sub]
        dec_tps = [r["decode_tok_per_s"] for r in sub]
        col = colors[idx % len(colors)]
        mk = markers[idx % len(markers)]
        ax1.plot(c, tot_tps, marker=mk, linewidth=2.2, color=col, label=scen)
        ax2.plot(c, dec_tps, marker=mk, linewidth=2.2, color=col, label=scen)

    ax1.set_title("Total Throughput (Prefill + Decode) vs. Concurrency")
    ax1.set_xlabel("Concurrent Agentic Sessions")
    ax1.set_ylabel("Total Throughput (tokens / sec)")
    ax1.legend(loc="best", fontsize=9)

    ax2.set_title("Output Decode Throughput vs. Concurrency")
    ax2.set_xlabel("Concurrent Agentic Sessions")
    ax2.set_ylabel("Decode Throughput (output tokens / sec)")
    ax2.legend(loc="best", fontsize=9)

    fig.suptitle("DeepSeek-V4.1-Flash (552B) on Multi-Host TPU v6e — Agentic Throughput Scaling", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = charts_dir / "throughput_vs_concurrency.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out}")


def plot_ttft_tpot_pareto(records: list[dict], charts_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    scenarios = sorted(set(r["scenario"] for r in records if r["scenario"].startswith("agentic_")))
    if not scenarios:
        scenarios = sorted(set(r["scenario"] for r in records))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "P"]

    for idx, scen in enumerate(scenarios):
        sub = sorted([r for r in records if r["scenario"] == scen], key=lambda x: x["concurrency"])
        if not sub:
            continue
        c = [r["concurrency"] for r in sub]
        ttft = [r["ttft_p50_ms"] / 1000.0 for r in sub]
        tpot = [r["tpot_p50_ms"] for r in sub]
        col = colors[idx % len(colors)]
        mk = markers[idx % len(markers)]
        ax1.plot(c, ttft, marker=mk, linewidth=2.2, color=col, label=scen)
        ax2.plot(c, tpot, marker=mk, linewidth=2.2, color=col, label=scen)

    ax1.set_title("Median Time-To-First-Token (TTFT P50) vs. Concurrency")
    ax1.set_xlabel("Concurrent Agentic Sessions")
    ax1.set_ylabel("TTFT P50 (seconds)")
    ax1.legend(loc="best", fontsize=9)

    ax2.set_title("Median Time-Per-Output-Token (TPOT P50) vs. Concurrency")
    ax2.set_xlabel("Concurrent Agentic Sessions")
    ax2.set_ylabel("TPOT P50 (ms / token)")
    ax2.legend(loc="best", fontsize=9)

    fig.suptitle("DeepSeek-V4.1-Flash on Multi-Host TPU v6e — Responsiveness & Interactivity", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = charts_dir / "latency_vs_concurrency.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out}")


def plot_optimization_ablation(records: list[dict], charts_dir: Path) -> None:
    # Pick the peak concurrency record for each scenario to show end-to-end optimization & context scaling
    by_scen: dict[str, dict] = {}
    for r in records:
        scen = r["scenario"]
        if scen not in by_scen or r["total_tok_per_s"] > by_scen[scen]["total_tok_per_s"]:
            by_scen[scen] = r
    ablation = sorted(by_scen.values(), key=lambda x: x["total_tok_per_s"])
    if not ablation:
        return

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    labels = [
        f"{r['scenario']}\n(C={r['concurrency']}, ISL={r['target_input_tokens'] // 1000}K)"
        for r in ablation
    ]
    x = np.arange(len(labels))
    width = 0.55

    tot_vals = [r["total_tok_per_s"] for r in ablation]
    bar_colors = [
        "#ff7f0e" if "bf16" in r["scenario"] else ("#2ca02c" if "prefix_cached" in r["scenario"] else "#1f77b4")
        for r in ablation
    ]
    bars = ax.bar(x, tot_vals, width, color=bar_colors)

    ax.set_title("Peak Cluster Throughput by Optimization & Context Length (24× TPU v6e Chips)")
    ax.set_ylabel("Total Throughput (tokens / sec)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:,.1f} tok/s",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylim(0, max(tot_vals) * 1.15)
    fig.tight_layout()
    out = charts_dir / "optimization_ablation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out}")


def plot_fp8_vs_bf16(records: list[dict], charts_dir: Path) -> None:
    fp8_recs = sorted(
        [r for r in records if "fp8" in r.get("scenario", "").lower() and "ablation" in r.get("scenario", "").lower()],
        key=lambda x: x["concurrency"],
    )
    bf16_recs = sorted(
        [r for r in records if "bf16" in r.get("scenario", "").lower() and "ablation" in r.get("scenario", "").lower()],
        key=lambda x: x["concurrency"],
    )
    if not fp8_recs or not bf16_recs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    c_fp8 = [r["concurrency"] for r in fp8_recs]
    c_bf16 = [r["concurrency"] for r in bf16_recs]

    ax1.plot(c_bf16, [r["total_tok_per_s"] for r in bf16_recs], marker="s", linewidth=2.4, color="#ff7f0e", label="BF16 KV Cache + Indexer (--kv-cache-dtype auto)")
    ax1.plot(c_fp8, [r["total_tok_per_s"] for r in fp8_recs], marker="o", linewidth=2.4, color="#2ca02c", label="FP8 KV Cache + Indexer (--kv-cache-dtype fp8)")
    ax1.set_title("Total Agentic Throughput: FP8 vs. BF16")
    ax1.set_xlabel("Concurrent Agentic Sessions")
    ax1.set_ylabel("Total Throughput (tokens / sec)")
    ax1.legend(loc="best", fontsize=9)

    ax2.plot(c_bf16, [r["tpot_p50_ms"] for r in bf16_recs], marker="s", linewidth=2.4, color="#ff7f0e", label="BF16 KV Cache + Indexer")
    ax2.plot(c_fp8, [r["tpot_p50_ms"] for r in fp8_recs], marker="o", linewidth=2.4, color="#2ca02c", label="FP8 KV Cache + Indexer")
    ax2.set_title("Decode Latency (TPOT P50): FP8 vs. BF16")
    ax2.set_xlabel("Concurrent Agentic Sessions")
    ax2.set_ylabel("TPOT P50 (ms / token)")
    ax2.legend(loc="best", fontsize=9)

    fig.suptitle("DeepSeek-V4.1-Flash on Multi-Host TPU v6e — FP8 vs. BF16 KV & Indexer Benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = charts_dir / "fp8_vs_bf16_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    charts_dir = args.results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    records = load_all_results(args.results_dir)
    if not records:
        raise SystemExit(f"No bench_*.json files found in {args.results_dir}")

    save_summary_csv(records, args.results_dir / "summary_metrics.csv")
    plot_throughput_vs_concurrency(records, charts_dir)
    plot_ttft_tpot_pareto(records, charts_dir)
    plot_optimization_ablation(records, charts_dir)
    plot_fp8_vs_bf16(records, charts_dir)


if __name__ == "__main__":
    main()
