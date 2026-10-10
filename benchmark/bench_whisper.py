#!/usr/bin/env python3
"""
Benchmark the Whisper STT server across batch sizes.

Every audio/video file in the test folder is sent to the server at every batch size,
`--repeats` times each. Runs are interleaved (repeat -> file -> batch size) so that a
slow period on the GPU is spread over all batch sizes instead of penalising one.

Example:
    python bench_whisper.py --folder test_folder --url http://localhost:8788 \
        --batch-sizes 1,2,4,8,16 --repeats 3 --language ja

Output (in bench_results/<timestamp>/):
    runs.csv        one row per request (written as the benchmark goes)
    summary.csv     mean / spread per file and batch size
    overall.csv     one row per batch size, all files combined
    summary.txt     the tables printed at the end
    transcripts/    the text of every run, for checking quality by eye

Needs: pip install requests          (optional: pip install rapidfuzz, for text similarity)
The server must be the updated whisper_server.py (it accepts `batch_size` and returns `metrics`).
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:  # optional, only used for the transcript-similarity column
    from rapidfuzz.distance import Levenshtein
except Exception:  # pragma: no cover
    Levenshtein = None

MEDIA_EXT = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
    ".mp4", ".mkv", ".webm", ".mov", ".avi",
}

RUN_FIELDS = [
    "timestamp", "file_id", "file", "file_mb", "audio_s", "batch_size", "repeat", "ok", "error",
    "wall_s", "infer_s", "decode_ms", "queue_wait_ms", "speed_x", "n_chunks", "s_per_chunk",
    "output_tokens", "tokens_per_s", "chars", "segments",
    "vram_peak_gb", "vram_reserved_gb", "gpu_util_avg", "gpu_util_max", "similarity_pct",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark the Whisper server across batch sizes")
    p.add_argument("--folder", default="test_folder", help="folder with audio/video files")
    p.add_argument("--url", default="http://localhost:8788", help="base URL of the Whisper server")
    p.add_argument("--batch-sizes", default="1,2,4,8,16", help="comma-separated batch sizes")
    p.add_argument("--repeats", type=int, default=3, help="runs per file and batch size")
    p.add_argument("--language", default="ja")
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    p.add_argument("--ref-batch", type=int, default=None,
                   help="batch size whose transcript is the reference for similarity "
                        "(default: the first one in --batch-sizes)")
    p.add_argument("--cooldown", type=float, default=2.0, help="seconds to wait between runs")
    p.add_argument("--timeout", type=float, default=3 * 3600, help="seconds allowed per request")
    p.add_argument("--out", default="bench_results", help="folder for the results")
    return p.parse_args()


# ----------------------------------------------------------------------------- helpers

def mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def stdev(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def vmax(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def vmin(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return min(vals) if vals else None


def fmt(value: Any, digits: int = 1) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(c.rjust(widths[i]) if i else c.ljust(widths[i]) for i, c in enumerate(r))
            for r in rows]
    return "\n".join([line, sep, *body])


def similarity(a: Optional[str], b: Optional[str]) -> Optional[float]:
    """Character-level similarity in percent (100 = identical). Needs rapidfuzz."""
    if Levenshtein is None or a is None or b is None:
        return None
    if not a and not b:
        return 100.0
    return round(Levenshtein.normalized_similarity(a, b) * 100, 2)


# ----------------------------------------------------------------------------- server calls

def check_server(url: str) -> None:
    try:
        resp = requests.get(f"{url}/health", timeout=10)
        resp.raise_for_status()
        info = resp.json()
    except Exception as exc:
        sys.exit(f"Cannot reach the server at {url}/health: {exc}")
    print(f"Server: model={info.get('model')} device={info.get('device')} "
          f"default_batch={info.get('batch_size')} max_batch={info.get('max_batch_size')}")
    if info.get("status") != "ok":
        sys.exit(f"Server is not ready: status={info.get('status')} error={info.get('error')}")
    if "max_batch_size" not in info:
        sys.exit("This server does not look like the updated whisper_server.py "
                 "(no max_batch_size in /health). Deploy the new server first.")
    if info.get("busy"):
        print("WARNING: the server is busy with another request. Timings will not be clean.")


def run_once(session: requests.Session, url: str, path: Path, upload_name: str,
             language: str, task: str, batch_size: int, timeout: float
             ) -> Tuple[Dict[str, Any], Optional[str]]:
    """Send one file once. Returns (measurements, transcript text or None)."""
    data = {"language": language, "task": task, "batch_size": str(batch_size)}
    t0 = time.perf_counter()
    try:
        with open(path, "rb") as fh:
            # ASCII upload name: avoids multipart filename-encoding issues with non-ASCII names
            resp = session.post(f"{url}/transcribe/timestamps", data=data,
                                files={"file": (upload_name, fh)}, timeout=timeout)
    except requests.RequestException as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "wall_s": round(time.perf_counter() - t0, 2)}, None
    wall_s = time.perf_counter() - t0

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = resp.text[:200]
        return {"ok": False, "error": f"HTTP {resp.status_code}: {detail}",
                "wall_s": round(wall_s, 2)}, None

    body = resp.json()
    m = body.get("metrics") or {}
    if m.get("batch_size") != batch_size:
        sys.exit("The server ignored the requested batch_size (response has no matching "
                 "metrics.batch_size). Deploy the updated whisper_server.py first.")

    audio_s = float(body.get("duration_s") or 0.0)
    infer_s = (m.get("infer_ms") or 0.0) / 1000.0
    tokens = m.get("output_tokens")
    n_chunks = m.get("n_chunks_est")
    text = body.get("text") or ""
    row = {
        "ok": True, "error": "",
        "audio_s": audio_s,
        "wall_s": round(wall_s, 2),
        "infer_s": round(infer_s, 2),
        "decode_ms": m.get("decode_ms"),
        "queue_wait_ms": m.get("queue_wait_ms"),
        "speed_x": round(audio_s / infer_s, 2) if infer_s > 0 else None,
        "n_chunks": n_chunks,
        "s_per_chunk": round(infer_s / n_chunks, 3) if n_chunks else None,
        "output_tokens": tokens,
        "tokens_per_s": round(tokens / infer_s, 1) if tokens is not None and infer_s > 0 else None,
        "chars": len(text),
        "segments": len(body.get("segments") or []),
        "vram_peak_gb": m.get("vram_peak_gb"),
        "vram_reserved_gb": m.get("vram_reserved_peak_gb"),
        "gpu_util_avg": m.get("gpu_util_avg"),
        "gpu_util_max": m.get("gpu_util_max"),
    }
    return row, text


# ----------------------------------------------------------------------------- summaries

def summarise(runs: List[Dict[str, Any]], batch_sizes: List[int]
              ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (per file x batch rows, per batch rows)."""
    ok = [r for r in runs if r.get("ok")]
    file_ids = sorted({r["file_id"] for r in runs})

    per_file: List[Dict[str, Any]] = []
    for fid in file_ids:
        for bs in batch_sizes:
            sel = [r for r in ok if r["file_id"] == fid and r["batch_size"] == bs]
            failed = sum(1 for r in runs
                         if r["file_id"] == fid and r["batch_size"] == bs and not r.get("ok"))
            if not sel and not failed:
                continue
            col = lambda k: [r.get(k) for r in sel]  # noqa: E731
            per_file.append({
                "file_id": fid,
                "audio_s": sel[0]["audio_s"] if sel else None,
                "batch_size": bs,
                "runs_ok": len(sel),
                "runs_failed": failed,
                "infer_s_mean": mean(col("infer_s")),
                "infer_s_std": stdev(col("infer_s")),
                "infer_s_min": vmin(col("infer_s")),
                "infer_s_max": vmax(col("infer_s")),
                "speed_x_mean": mean(col("speed_x")),
                "s_per_chunk_mean": mean(col("s_per_chunk")),
                "output_tokens_mean": mean(col("output_tokens")),
                "tokens_per_s_mean": mean(col("tokens_per_s")),
                "chars_mean": mean(col("chars")),
                "segments_mean": mean(col("segments")),
                "vram_peak_gb_max": vmax(col("vram_peak_gb")),
                "vram_reserved_gb_max": vmax(col("vram_reserved_gb")),
                "gpu_util_avg_mean": mean(col("gpu_util_avg")),
                "gpu_util_max": vmax(col("gpu_util_max")),
                "similarity_pct_mean": mean(col("similarity_pct")),
            })

    # Overall: only files that succeeded at every batch size, so the comparison is like for like
    done = {(s["file_id"], s["batch_size"]) for s in per_file if s["runs_ok"]}
    # (a batch size that failed on every file, e.g. out of memory, is left out)
    ok_batches = [bs for bs in batch_sizes if any((fid, bs) in done for fid in file_ids)]
    common = [fid for fid in file_ids if all((fid, bs) in done for bs in ok_batches)]
    overall: List[Dict[str, Any]] = []
    for bs in ok_batches:
        sel = [s for s in per_file if s["batch_size"] == bs and s["file_id"] in common]
        if not sel:
            continue
        audio = sum(s["audio_s"] for s in sel)
        infer = sum(s["infer_s_mean"] for s in sel)
        tokens = [s["output_tokens_mean"] for s in sel]
        overall.append({
            "batch_size": bs,
            "files": len(sel),
            "audio_s_total": round(audio, 1),
            "infer_s_total": round(infer, 1),
            "speed_x": round(audio / infer, 2) if infer > 0 else None,
            "tokens_per_s": round(sum(tokens) / infer, 1)
                            if infer > 0 and all(t is not None for t in tokens) else None,
            "vram_peak_gb_max": vmax([s["vram_peak_gb_max"] for s in sel]),
            "vram_reserved_gb_max": vmax([s["vram_reserved_gb_max"] for s in sel]),
            "gpu_util_avg_mean": mean([s["gpu_util_avg_mean"] for s in sel]),
            "similarity_pct_mean": mean([s["similarity_pct_mean"] for s in sel]),
            "spread_pct_worst": vmax([
                100 * s["infer_s_std"] / s["infer_s_mean"]
                for s in sel if s["infer_s_mean"] and s["infer_s_std"] is not None
            ]),
        })
    fastest = vmin([o["infer_s_total"] for o in overall])
    for o in overall:
        o["time_vs_fastest_pct"] = (
            round(100 * (o["infer_s_total"] / fastest - 1), 1) if fastest else None
        )
    return per_file, overall


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})


def build_report(files: List[Tuple[str, Path]], per_file: List[Dict[str, Any]],
                 overall: List[Dict[str, Any]], args: argparse.Namespace, ref_batch: int) -> str:
    parts: List[str] = []
    parts.append(f"Whisper batch-size benchmark  |  {datetime.now():%Y-%m-%d %H:%M}")
    parts.append(f"repeats={args.repeats}  language={args.language}  task={args.task}  "
                 f"similarity reference = batch {ref_batch}")
    parts.append("")
    parts.append("Files")
    audio = {s["file_id"]: s["audio_s"] for s in per_file if s["audio_s"]}
    parts.append(render_table(
        ["id", "audio", "name"],
        [[fid, f"{audio[fid] / 60:.1f} min" if fid in audio else "-", p.name] for fid, p in files],
    ))

    parts.append("")
    parts.append("Per file and batch size (means over the repeats)")
    parts.append(render_table(
        ["file", "batch", "ok", "fail", "time s", "+/- s", "speed x", "s/chunk",
         "tokens", "tok/s", "segs", "peak GB", "resv GB", "GPU %", "similar %"],
        [[s["file_id"], str(s["batch_size"]), str(s["runs_ok"]), str(s["runs_failed"]),
          fmt(s["infer_s_mean"]), fmt(s["infer_s_std"]), fmt(s["speed_x_mean"]),
          fmt(s["s_per_chunk_mean"], 2), fmt(s["output_tokens_mean"], 0),
          fmt(s["tokens_per_s_mean"]), fmt(s["segments_mean"], 0),
          fmt(s["vram_peak_gb_max"]), fmt(s["vram_reserved_gb_max"]),
          fmt(s["gpu_util_avg_mean"], 0), fmt(s["similarity_pct_mean"])]
         for s in per_file],
    ))

    parts.append("")
    parts.append("Overall per batch size (files that completed at every batch size that worked)")
    if overall:
        parts.append(render_table(
            ["batch", "files", "audio s", "time s", "speed x", "vs fastest", "tok/s",
             "peak GB", "resv GB", "GPU %", "similar %", "worst spread"],
            [[str(o["batch_size"]), str(o["files"]), fmt(o["audio_s_total"]),
              fmt(o["infer_s_total"]), fmt(o["speed_x"]),
              "fastest" if o["time_vs_fastest_pct"] == 0 else f"+{fmt(o['time_vs_fastest_pct'])}%",
              fmt(o["tokens_per_s"]), fmt(o["vram_peak_gb_max"]), fmt(o["vram_reserved_gb_max"]),
              fmt(o["gpu_util_avg_mean"], 0), fmt(o["similarity_pct_mean"]),
              f"{fmt(o['spread_pct_worst'])}%"]
             for o in overall],
        ))
    else:
        parts.append("(no file completed at every batch size)")

    parts.append("")
    parts.append("Columns: time = model time only (no upload, decoding or queueing). "
                 "speed x = audio length / time. peak = memory in use, resv = memory held "
                 "(what nvidia-smi shows). GPU % covers the whole card. similar % = transcript "
                 "match against the reference batch (its own repeats show run-to-run variation). "
                 "worst spread = largest std/mean of the timings.")
    if Levenshtein is None:
        parts.append("Similarity is empty because rapidfuzz is not installed (pip install rapidfuzz).")
    return "\n".join(parts)


# ----------------------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    url = args.url.rstrip("/")
    try:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    except ValueError:
        sys.exit(f"--batch-sizes must be comma-separated integers, got {args.batch_sizes!r}")
    if not batch_sizes or args.repeats < 1:
        sys.exit("Need at least one batch size and one repeat")
    ref_batch = args.ref_batch if args.ref_batch is not None else batch_sizes[0]
    if ref_batch not in batch_sizes:
        sys.exit(f"--ref-batch {ref_batch} is not in --batch-sizes")

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXT)
    if not paths:
        sys.exit(f"No audio/video files in {folder}")
    files = [(f"F{i + 1}", p) for i, p in enumerate(paths)]

    check_server(url)

    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    total = len(files) * len(batch_sizes) * args.repeats
    print(f"{len(files)} file(s) x {len(batch_sizes)} batch size(s) x {args.repeats} repeat(s) "
          f"= {total} runs. Results: {out_dir}")
    for fid, p in files:
        print(f"  {fid} = {p.name} ({p.stat().st_size / 1e6:.1f} MB)")

    runs: List[Dict[str, Any]] = []
    texts: Dict[Tuple[str, int, int], str] = {}
    session = requests.Session()
    done = 0
    interrupted = False

    with open(out_dir / "runs.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        try:
            for repeat in range(1, args.repeats + 1):
                for fid, path in files:
                    for bs in batch_sizes:
                        done += 1
                        print(f"[{done}/{total}] {fid} batch={bs} repeat={repeat} ... ",
                              end="", flush=True)
                        row, text = run_once(session, url, path, f"bench_{fid}{path.suffix}",
                                             args.language, args.task, bs, args.timeout)
                        row.update(
                            timestamp=datetime.now().isoformat(timespec="seconds"),
                            file_id=fid, file=path.name,
                            file_mb=round(path.stat().st_size / 1e6, 1),
                            batch_size=bs, repeat=repeat,
                        )
                        if row["ok"]:
                            texts[(fid, bs, repeat)] = text or ""
                            (out_dir / "transcripts" / f"{fid}_b{bs}_r{repeat}.txt").write_text(
                                text or "", encoding="utf-8")
                            print(f"{row['infer_s']:.1f}s ({row['speed_x']}x), "
                                  f"peak {fmt(row['vram_peak_gb'])} GB, "
                                  f"GPU {fmt(row['gpu_util_avg'], 0)}%, "
                                  f"{fmt(row['output_tokens'])} tokens")
                        else:
                            print(f"FAILED: {row['error']}")
                        runs.append(row)
                        writer.writerow({k: ("" if row.get(k) is None else row.get(k))
                                         for k in RUN_FIELDS})
                        fh.flush()
                        time.sleep(args.cooldown)
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted - summarising the runs completed so far.")

    # Transcript similarity against the reference batch (first successful repeat of it)
    for fid, _ in files:
        ref_key = next((k for k in sorted(texts) if k[0] == fid and k[1] == ref_batch), None)
        if ref_key is None:
            continue
        for row in runs:
            key = (row["file_id"], row["batch_size"], row["repeat"])
            if row["file_id"] == fid and row.get("ok") and key != ref_key:
                row["similarity_pct"] = similarity(texts.get(key), texts[ref_key])
    write_csv(out_dir / "runs.csv", runs, RUN_FIELDS)  # rewrite with the similarity column

    per_file, overall = summarise(runs, batch_sizes)
    write_csv(out_dir / "summary.csv", per_file)
    write_csv(out_dir / "overall.csv", overall)
    report = build_report(files, per_file, overall, args, ref_batch)
    (out_dir / "summary.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    print(f"\nSaved to {out_dir}" + (" (partial: interrupted)" if interrupted else ""))


if __name__ == "__main__":
    main()
