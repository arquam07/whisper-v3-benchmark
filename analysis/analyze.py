#!/usr/bin/env python3
"""
Rebuild the summary tables and charts from a runs.csv written by bench_whisper.py.

    python analysis/analyze.py results/2026-10-08/runs.csv --assets assets

Writes summary.csv and overall.csv next to runs.csv, and three PNG charts to --assets.
Failed runs are counted but excluded from every average.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

# Light-surface palette (fixed order: series 1, 2, 3)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"


def style() -> None:
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": "Inter" if "Inter" in have else "DejaVu Sans",
        "font.size": 11,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "ytick.major.size": 0, "xtick.major.size": 0,
        "legend.frameon": False,
    })


def titled(fig, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.965, title, fontsize=14, fontweight="bold", ha="left", va="top")
    fig.text(0.02, 0.905, subtitle, fontsize=10.5, color=INK2, ha="left", va="top")


def summarise(runs: pd.DataFrame):
    ok = runs[runs["ok"]]
    failed = runs[~runs["ok"]].groupby(["file_id", "batch_size"]).size().rename("runs_failed")
    g = ok.groupby(["file_id", "batch_size"])
    summary = g.agg(
        audio_s=("audio_s", "first"),
        runs_ok=("infer_s", "size"),
        infer_s_mean=("infer_s", "mean"),
        infer_s_std=("infer_s", "std"),
        infer_s_min=("infer_s", "min"),
        infer_s_max=("infer_s", "max"),
        speed_x_mean=("speed_x", "mean"),
        s_per_chunk_mean=("s_per_chunk", "mean"),
        output_tokens_mean=("output_tokens", "mean"),
        tokens_per_s_mean=("tokens_per_s", "mean"),
        segments_mean=("segments", "mean"),
        vram_peak_gb_max=("vram_peak_gb", "max"),
        vram_reserved_gb_max=("vram_reserved_gb", "max"),
        gpu_util_avg_mean=("gpu_util_avg", "mean"),
        similarity_pct_mean=("similarity_pct", "mean"),
    ).join(failed).fillna({"runs_failed": 0}).reset_index()
    summary["runs_failed"] = summary["runs_failed"].astype(int)
    summary["spread_pct"] = 100 * summary["infer_s_std"] / summary["infer_s_mean"]

    o = summary.groupby("batch_size").agg(
        files=("file_id", "size"),
        runs_ok=("runs_ok", "sum"),
        audio_s_total=("audio_s", "sum"),
        infer_s_total=("infer_s_mean", "sum"),
        tokens_total=("output_tokens_mean", "sum"),
        vram_peak_gb_max=("vram_peak_gb_max", "max"),
        vram_reserved_gb_max=("vram_reserved_gb_max", "max"),
        gpu_util_avg_mean=("gpu_util_avg_mean", "mean"),
        similarity_pct_mean=("similarity_pct_mean", "mean"),
        spread_pct_worst=("spread_pct", "max"),
    ).reset_index()
    o["speed_x"] = o["audio_s_total"] / o["infer_s_total"]
    o["tokens_per_s"] = o["tokens_total"] / o["infer_s_total"]
    o["time_vs_fastest_pct"] = 100 * (o["infer_s_total"] / o["infer_s_total"].min() - 1)
    return summary, o.drop(columns="tokens_total")


def chart_tradeoff(o: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(top=0.80, left=0.09, right=0.96, bottom=0.13)
    titled(fig, "Batch 8 is the fastest setting; batch 16 costs more memory and is slower",
           "Overall speed vs. GPU memory held, one point per batch size (3 recordings, 125 min of audio)")
    x, y = o["vram_reserved_gb_max"], o["speed_x"]
    ax.plot(x, y, color=SERIES[0], lw=2, zorder=2)
    ax.scatter(x, y, s=70, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    offsets = {1: (10, -4), 2: (10, -6), 4: (-8, 10), 8: (0, 12), 16: (0, 12)}
    for bs, xi, yi in zip(o["batch_size"], x, y):
        dx, dy = offsets.get(int(bs), (0, 12))
        ax.annotate(f"batch {int(bs)}\n{yi:.1f}x", (xi, yi), xytext=(dx, dy),
                    textcoords="offset points", fontsize=10, color=INK,
                    ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                    va="bottom" if dy > 0 else "top")
    ax.set_xlabel("GPU memory reserved (GiB)")
    ax.set_ylabel("Speed (x realtime)")
    ax.set_xlim(0, 36)
    ax.set_ylim(6, 16.5)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def chart_speed(summary: pd.DataFrame, labels: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(top=0.80, left=0.09, right=0.80, bottom=0.13)
    titled(fig, "Every recording peaks at batch 4-8 and slows down at 16",
           "Mean speed per recording and batch size (model time only)")
    batches = sorted(summary["batch_size"].unique())
    pos = {b: i for i, b in enumerate(batches)}
    ends = []
    for color, (fid, grp) in zip(SERIES, summary.groupby("file_id")):
        grp = grp.sort_values("batch_size")
        xs = [pos[b] for b in grp["batch_size"]]
        ax.plot(xs, grp["speed_x_mean"], color=color, lw=2, marker="o", ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=labels[fid])
        ends.append((grp["speed_x_mean"].iloc[-1], xs[-1], labels[fid]))
    # Direct labels at the line ends, pushed apart so near-identical series stay readable
    placed = None
    for y, x, label in sorted(ends):
        y_text = y if placed is None else max(y, placed + 1.3)
        ax.annotate(label, (x, y), xytext=(x + 0.12, y_text), textcoords="data",
                    fontsize=10, color=INK, va="center", annotation_clip=False)
        placed = y_text
    ax.set_xticks(range(len(batches)), [str(b) for b in batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speed (x realtime)")
    ax.set_ylim(0, 22)
    ax.set_yticks(range(0, 21, 5))
    ax.legend(loc="upper left", ncols=3, fontsize=9.5, handlelength=1.6,
              bbox_to_anchor=(0, 1.02), borderaxespad=0)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def chart_vram(o: pd.DataFrame, fit: tuple, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(top=0.80, left=0.09, right=0.96, bottom=0.13)
    slope, base = fit
    titled(fig, "Memory grows linearly with batch size",
           f"Peak in use is about {base:.1f} GiB for the model plus {slope:.2f} GiB per chunk in the batch")
    idx = np.arange(len(o))
    w = 0.36
    for k, (col, label) in enumerate([("vram_peak_gb_max", "Peak in use"),
                                      ("vram_reserved_gb_max", "Reserved by PyTorch (peak)")]):
        xs = idx + (k - 0.5) * (w + 0.03)
        ax.bar(xs, o[col], width=w, color=SERIES[k], label=label)
        for xi, v in zip(xs, o[col]):
            ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(idx, [str(int(b)) for b in o["batch_size"]])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("GPU memory (GiB)")
    ax.set_ylim(0, 36)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs_csv")
    ap.add_argument("--assets", default="assets", help="folder for the PNG charts")
    args = ap.parse_args()

    runs_path = Path(args.runs_csv)
    runs = pd.read_csv(runs_path, encoding="utf-8-sig")
    runs["ok"] = runs["ok"].astype(str).str.lower() == "true"
    summary, overall = summarise(runs)
    summary.round(3).to_csv(runs_path.with_name("summary.csv"), index=False)
    overall.round(3).to_csv(runs_path.with_name("overall.csv"), index=False)

    slope, base = np.polyfit(overall["batch_size"], overall["vram_peak_gb_max"], 1)
    labels = {
        fid: f"{Path(name).stem.replace('_', ' ')} ({audio / 60:.0f} min)"
        for fid, name, audio in runs[runs["ok"]][["file_id", "file", "audio_s"]]
        .drop_duplicates().itertuples(index=False)
    }

    style()
    assets = Path(args.assets)
    assets.mkdir(parents=True, exist_ok=True)
    chart_tradeoff(overall, assets / "tradeoff.png")
    chart_speed(summary, labels, assets / "speed_by_batch.png")
    chart_vram(overall, (slope, base), assets / "vram_by_batch.png")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(f"runs: {int(runs['ok'].sum())} ok, {int((~runs['ok']).sum())} failed")
    print(f"peak VRAM fit: {base:.2f} GiB + {slope:.3f} GiB x batch_size\n")
    print(overall.round(2).to_string(index=False))
    print()
    print(summary[["file_id", "batch_size", "runs_ok", "runs_failed", "infer_s_mean", "speed_x_mean",
                   "s_per_chunk_mean", "output_tokens_mean", "segments_mean", "gpu_util_avg_mean",
                   "similarity_pct_mean", "spread_pct"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
