"""Merge outputs from parallel chunk processing into a single run directory.

Combines per-chunk CSVs, overlay videos, and contact analysis outputs.

Usage:
    python scripts/merge_chunks.py outputs/runs/*_chunk*/

    # Custom output directory:
    python scripts/merge_chunks.py outputs/runs/*_chunk*/ -o outputs/runs/merged
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_chunk_dirs(paths: List[str]) -> List[Path]:
    """Sort chunk directories by chunk ID extracted from directory name."""
    dirs = []
    for p in paths:
        d = Path(p)
        if d.is_dir():
            dirs.append(d)
    # Sort by chunk ID (e.g., "2026-02-26_120000_sam2_yolo_chunk0" → 0)
    def chunk_key(d: Path) -> int:
        name = d.name
        if "_chunk" in name:
            try:
                return int(name.rsplit("_chunk", 1)[1])
            except ValueError:
                pass
        return 0
    dirs.sort(key=chunk_key)
    return dirs


def merge_csvs(chunk_dirs: List[Path], filename: str, output_dir: Path) -> Path | None:
    """Concatenate a CSV file from each chunk, keeping header from first chunk."""
    out_path = output_dir / filename
    header_written = False
    total_rows = 0

    with out_path.open("w", newline="", encoding="utf-8") as out_f:
        for cdir in chunk_dirs:
            csv_path = cdir / filename
            if not csv_path.exists():
                # Check in contacts/ subdirectory
                csv_path = cdir / "contacts" / filename
                if not csv_path.exists():
                    continue
            with csv_path.open("r", encoding="utf-8") as in_f:
                reader = csv.reader(in_f)
                for i, row in enumerate(reader):
                    if i == 0:
                        if not header_written:
                            csv.writer(out_f).writerow(row)
                            header_written = True
                        continue
                    csv.writer(out_f).writerow(row)
                    total_rows += 1

    if total_rows == 0:
        out_path.unlink(missing_ok=True)
        return None
    logger.info("Merged %s: %d rows from %d chunks", filename, total_rows, len(chunk_dirs))
    return out_path


def merge_overlay_videos(chunk_dirs: List[Path], output_dir: Path) -> Path | None:
    """Concatenate overlay videos using ffmpeg (if available) or OpenCV."""
    video_paths = []
    for cdir in chunk_dirs:
        overlays = cdir / "overlays"
        if overlays.exists():
            # Find any video file in overlays/ (supports new naming: pipeline_model_date.avi)
            found = sorted(overlays.glob("*.avi")) + sorted(overlays.glob("*.mp4"))
            if found:
                video_paths.append(found[0])  # take the first video found

    if not video_paths:
        logger.warning("No overlay videos found to merge")
        return None

    # Always use _merged suffix so run_parallel.sh can find the video
    ext = video_paths[0].suffix
    base_name = video_paths[0].stem

    if len(video_paths) == 1:
        out_path = output_dir / "overlays" / f"{base_name}_merged{ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_paths[0], out_path)
        return out_path
    out_path = output_dir / "overlays" / f"{base_name}_merged{ext}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write concat list file
    list_file = output_dir / "overlays" / "_concat_list.txt"
    with list_file.open("w") as f:
        for vp in video_paths:
            f.write(f"file '{vp.resolve()}'\n")

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(list_file), "-c", "copy", str(out_path)],
            check=True, capture_output=True, text=True,
        )
        list_file.unlink()
        logger.info("Merged %d overlay videos → %s (ffmpeg)", len(video_paths), out_path)
        return out_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("ffmpeg not available or failed. Falling back to OpenCV concat.")
        list_file.unlink(missing_ok=True)

    # Fallback: OpenCV concat (re-encodes)
    import cv2
    first = cv2.VideoCapture(str(video_paths[0]))
    fps = first.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first.release()

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_path = out_path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
    writer.release()
    logger.info("Merged %d overlay videos → %s (OpenCV)", len(video_paths), out_path)
    return out_path


def merge_session_summaries(chunk_dirs: List[Path], output_dir: Path) -> None:
    """Merge session_summary.json files from contact analysis chunks."""
    summaries = []
    for cdir in chunk_dirs:
        sp = cdir / "contacts" / "session_summary.json"
        if sp.exists():
            with sp.open() as f:
                summaries.append(json.load(f))

    if not summaries:
        return

    # Metadata from first chunk
    first_meta = summaries[0].get("metadata", {})
    total_frames = sum(s.get("metadata", {}).get("total_frames", 0) for s in summaries)
    fps = first_meta.get("fps", 30.0)
    duration_sec = total_frames / fps if fps > 0 else 0

    # Zone summary: sum frame counts, recalculate percentages
    zone_summary = {}
    for z in ["contact", "proximity", "independent"]:
        count = sum(s.get("zone_summary", {}).get(f"{z}_frames", 0) for s in summaries)
        zone_summary[f"{z}_frames"] = count
        zone_summary[f"{z}_pct"] = round(count / max(total_frames, 1) * 100, 2)

    # Contact type summary: sum bouts, durations, frames
    all_types = set()
    for s in summaries:
        all_types.update(s.get("contact_type_summary", {}).keys())

    type_summary = {}
    for ct in sorted(all_types):
        chunks = [s.get("contact_type_summary", {}).get(ct, {}) for s in summaries]
        total_bouts = sum(c.get("total_bouts", 0) for c in chunks)
        total_dur = sum(c.get("total_duration_sec", 0) for c in chunks)
        total_fr = sum(c.get("total_frames", 0) for c in chunks)
        entry = {
            "total_bouts": total_bouts,
            "total_duration_sec": round(total_dur, 2),
            "total_frames": total_fr,
            "pct_of_session": round(total_fr / max(total_frames, 1) * 100, 2),
        }
        # Merge by_investigator breakdowns
        by_inv_merged: dict = {}
        for c in chunks:
            for inv_key, inv_val in c.get("by_investigator", {}).items():
                if inv_key not in by_inv_merged:
                    by_inv_merged[inv_key] = {"bouts": 0, "duration_sec": 0.0}
                by_inv_merged[inv_key]["bouts"] += inv_val.get("bouts", 0)
                by_inv_merged[inv_key]["duration_sec"] = round(
                    by_inv_merged[inv_key]["duration_sec"] + inv_val.get("duration_sec", 0), 2
                )
        if by_inv_merged:
            entry["by_investigator"] = by_inv_merged
        type_summary[ct] = entry

    # Per-pair summary: sum across chunks
    all_pairs = set()
    for s in summaries:
        all_pairs.update(s.get("per_pair_summary", {}).keys())
    pair_summary = {}
    for pair_key in sorted(all_pairs):
        pair_total = 0.0
        by_type: dict = {}
        for s in summaries:
            pi = s.get("per_pair_summary", {}).get(pair_key, {})
            pair_total += pi.get("total_contact_sec", 0)
            for ct_name, ct_info in pi.get("contact_types", {}).items():
                if ct_name not in by_type:
                    by_type[ct_name] = {"bouts": 0, "duration_sec": 0.0}
                by_type[ct_name]["bouts"] += ct_info.get("bouts", 0)
                by_type[ct_name]["duration_sec"] = round(
                    by_type[ct_name]["duration_sec"] + ct_info.get("duration_sec", 0), 2
                )
        pair_summary[pair_key] = {
            "total_contact_sec": round(pair_total, 2),
            "total_contact_pct": round(pair_total / duration_sec * 100, 2) if duration_sec > 0 else 0.0,
            "contact_types": by_type,
        }

    # Quality: sum frame counts
    quality = {}
    for qk in ["high_mask_overlap_frames", "missing_keypoints_frames", "single_detection_frames"]:
        quality[qk] = sum(s.get("quality", {}).get(qk, 0) for s in summaries)
    flagged = sum(quality.values())
    quality["total_flagged_pct"] = round(flagged / max(total_frames, 1) * 100, 2)

    merged = {
        "metadata": {
            "video_path": first_meta.get("video_path", ""),
            "video_duration_sec": round(duration_sec, 2),
            "total_frames": total_frames,
            "fps": fps,
            "num_rats": first_meta.get("num_rats", 2),
            "analysis_date": datetime.now().isoformat(timespec="seconds"),
            "merged_from_chunks": len(summaries),
        },
        "zone_summary": zone_summary,
        "contact_type_summary": type_summary,
        "per_pair_summary": pair_summary,
        "quality": quality,
    }

    contacts_dir = output_dir / "contacts"
    contacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = contacts_dir / "session_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    logger.info("Merged session summary from %d chunks → %s", len(summaries), out_path)


def generate_merged_report(contacts_dir: Path) -> None:
    """Generate report.pdf from merged contact CSVs and session summary."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        import pandas as pd
    except ImportError:
        logger.warning("matplotlib/pandas not available — skipping PDF report")
        return

    csv_path = contacts_dir / "contacts_per_frame.csv"
    bouts_path = contacts_dir / "contact_bouts.csv"
    summary_path = contacts_dir / "session_summary.json"

    if not csv_path.exists():
        logger.warning("No contacts_per_frame.csv found — skipping report")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    bouts_df = pd.read_csv(bouts_path) if bouts_path.exists() else pd.DataFrame()
    summary = {}
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)

    ct_names = ["N2N", "N2AG", "N2B", "T2T", "FOL", "SBS"]
    ct_colors = {
        "N2N": "#1f77b4", "N2AG": "#ff7f0e", "N2B": "#2ca02c",
        "T2T": "#8c564b", "SBS": "#9467bd", "FOL": "#d62728",
    }
    meta = summary.get("metadata", {})
    duration = meta.get("video_duration_sec", df["time_sec"].max())
    fps = meta.get("fps", 30.0)

    pdf_path = contacts_dir / "report.pdf"
    with PdfPages(str(pdf_path)) as pdf:

        # ── Page 1: Timeline Ethogram ──
        if not bouts_df.empty and "start_time_sec" in bouts_df.columns:
            fig, ax = plt.subplots(figsize=(14, 3))
            for _, bout in bouts_df.iterrows():
                color = ct_colors.get(bout.get("contact_type", ""), "#888888")
                dur = bout.get("duration_sec", bout.get("end_time_sec", 0) - bout.get("start_time_sec", 0))
                ax.barh(0, dur, left=bout["start_time_sec"],
                        height=0.6, color=color, edgecolor="none")
            ax.set_yticks([0])
            ax.set_yticklabels(["Pair 0-1"])
            ax.set_xlabel("Time (seconds)")
            ax.set_title("Contact Ethogram (merged)")
            patches = [mpatches.Patch(color=c, label=t) for t, c in ct_colors.items()]
            ax.legend(handles=patches, loc="upper right", ncol=5, fontsize=8)
            ax.set_xlim(0, duration)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ── Page 2: Duration Histograms ──
        if not bouts_df.empty and "duration_sec" in bouts_df.columns:
            n_ct = len(ct_names)
            n_total = n_ct + 1  # +1 for "ALL"
            n_cols_hist = 3
            n_rows_hist = math.ceil(n_total / n_cols_hist)
            fig, axes = plt.subplots(n_rows_hist, n_cols_hist,
                                     figsize=(12, 4 * n_rows_hist), squeeze=False)
            all_durations = []
            for idx, ct in enumerate(ct_names):
                r, c = divmod(idx, n_cols_hist)
                ax = axes[r][c]
                durations = bouts_df[bouts_df["contact_type"] == ct]["duration_sec"].tolist()
                all_durations.extend(durations)
                if durations:
                    ax.hist(durations, bins=min(20, max(5, len(durations))),
                            color=ct_colors[ct], edgecolor="white")
                ax.set_title(ct)
                ax.set_xlabel("Duration (s)")
                ax.set_ylabel("Count")
            # "ALL" in next available cell
            r, c = divmod(n_ct, n_cols_hist)
            ax = axes[r][c]
            if all_durations:
                ax.hist(all_durations, bins=min(20, max(5, len(all_durations))),
                        color="#888888", edgecolor="white")
            ax.set_title("ALL")
            ax.set_xlabel("Duration (s)")
            ax.set_ylabel("Count")
            # Hide unused subplots
            for idx in range(n_total, n_rows_hist * n_cols_hist):
                r, c = divmod(idx, n_cols_hist)
                axes[r][c].set_visible(False)
            plt.suptitle("Bout Duration Distributions", fontsize=13)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ── Page 3: Pie Chart ──
        type_sums = summary.get("contact_type_summary", {})
        labels, sizes, colors = [], [], []
        for ct in ct_names:
            dur = type_sums.get(ct, {}).get("total_duration_sec", 0)
            if dur > 0:
                labels.append(ct)
                sizes.append(dur)
                colors.append(ct_colors[ct])
        fig, ax = plt.subplots(figsize=(8, 6))
        if sizes:
            ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
            ax.set_title("Contact Time by Type")
        else:
            ax.text(0.5, 0.5, "No contacts detected", ha="center", va="center", fontsize=14)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 4: Summary Table ──
        pair_summary = summary.get("per_pair_summary", {})
        if pair_summary:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.axis("off")
            table_data = []
            headers = ["Pair", "Total (s)"] + ct_names
            for pair_key, pair_info in pair_summary.items():
                row = [pair_key.replace("pair_", "")]
                row.append(f"{pair_info['total_contact_sec']:.1f}")
                for ct in ct_names:
                    ct_info = pair_info.get("contact_types", {}).get(ct, {})
                    dur_val = ct_info.get("duration_sec", 0)
                    bouts_val = ct_info.get("bouts", 0)
                    row.append(f"{dur_val:.1f}s ({bouts_val}b)")
                table_data.append(row)
            if table_data:
                table = ax.table(cellText=table_data, colLabels=headers,
                                 loc="center", cellLoc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
            ax.set_title("Per-Pair Contact Summary", fontsize=13, pad=20)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ── Page 5: Distance Over Time ──
        if "time_sec" in df.columns:
            fig, ax = plt.subplots(figsize=(14, 4))
            zone_colors_bg = {"contact": "#2ca02c", "proximity": "#ffcc00", "independent": "#dddddd"}
            if "zone" in df.columns:
                zones = df["zone"].values
                times = df["time_sec"].values
                i = 0
                while i < len(zones):
                    z = zones[i]
                    j = i
                    while j < len(zones) and zones[j] == z:
                        j += 1
                    ax.axvspan(times[i], times[min(j, len(times) - 1)],
                               color=zone_colors_bg.get(z, "#dddddd"), alpha=0.3, linewidth=0)
                    i = j
            if "nose_nose_dist_bl" in df.columns:
                valid = df.dropna(subset=["nose_nose_dist_bl"])
                ax.plot(valid["time_sec"], valid["nose_nose_dist_bl"],
                        color="#1f77b4", linewidth=0.5, alpha=0.8, label="Nose-Nose (BL)")
            if "centroid_dist_bl" in df.columns:
                valid = df.dropna(subset=["centroid_dist_bl"])
                ax.plot(valid["time_sec"], valid["centroid_dist_bl"],
                        color="#ff7f0e", linewidth=0.5, alpha=0.8, label="Centroid (BL)")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Distance (body lengths)")
            ax.set_title("Inter-Rat Distance Over Time")
            line_handles = ax.get_legend_handles_labels()[0]
            zone_patches = [mpatches.Patch(color=c, alpha=0.3, label=z.capitalize())
                            for z, c in zone_colors_bg.items()]
            ax.legend(handles=line_handles + zone_patches,
                      loc="upper right", fontsize=7, ncol=2)
            ax.set_xlim(0, df["time_sec"].max())
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ── Page 6: Investigator Role Breakdown ──
        if not bouts_df.empty and "investigator_slot" in bouts_df.columns:
            asym_types = ["N2AG", "N2B", "FOL"]
            inv_data = {}
            for ct_val in asym_types:
                ct_rows = bouts_df[bouts_df["contact_type"] == ct_val]
                slot_counts = ct_rows["investigator_slot"].dropna().astype(int).value_counts().to_dict()
                inv_data[ct_val] = slot_counts
            if any(inv_data.values()):
                fig, ax = plt.subplots(figsize=(10, 5))
                y_pos = list(range(len(asym_types)))
                slot_ids = sorted({s for sc in inv_data.values() for s in sc.keys()})
                slot_colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
                for idx, slot_id in enumerate(slot_ids):
                    vals = [inv_data[ct].get(slot_id, 0) for ct in asym_types]
                    color = slot_colors_list[idx % len(slot_colors_list)]
                    left = [0] * len(asym_types)
                    if idx > 0:
                        for prev in range(idx):
                            prev_slot = slot_ids[prev]
                            for j_idx, ct in enumerate(asym_types):
                                left[j_idx] += inv_data[ct].get(prev_slot, 0)
                    ax.barh(y_pos, vals, left=left, height=0.5,
                            color=color, label=f"Rat {slot_id}", edgecolor="white")
                ax.set_yticks(y_pos)
                ax.set_yticklabels(asym_types)
                ax.set_xlabel("Number of Bouts")
                ax.set_title("Investigator Role Breakdown (who initiates)")
                ax.legend(loc="lower right", fontsize=9)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        # ── Page 7: Cumulative Contact Time ──
        if "contact_type" in df.columns:
            fig, ax = plt.subplots(figsize=(14, 5))
            dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
            for ct in ct_names:
                ct_frames = df[df["contact_type"] == ct]
                if ct_frames.empty:
                    continue
                times = ct_frames["time_sec"].values
                cumulative = np.arange(1, len(times) + 1) * dt
                ax.plot(times, cumulative, color=ct_colors[ct], linewidth=1.5, label=ct)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Cumulative contact time (seconds)")
            ax.set_title("Cumulative Contact Time by Type")
            ax.legend(loc="upper left", fontsize=9)
            ax.set_xlim(0, df["time_sec"].max())
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    logger.info("Merged report: %s", pdf_path)


def _renumber_bout_ids(chunk_dirs: List[Path], contacts_dir: Path) -> None:
    """Renumber bout_ids in merged CSVs to avoid collisions across chunks.

    Each chunk starts bout_id at 0, so after naive concatenation there are
    duplicate IDs.  This function assigns globally unique bout_ids by adding
    an offset per chunk.
    """
    import pandas as pd

    bouts_path = contacts_dir / "contact_bouts.csv"
    pf_path = contacts_dir / "contacts_per_frame.csv"

    if not bouts_path.exists():
        return

    bouts_df = pd.read_csv(bouts_path)
    if bouts_df.empty or "bout_id" not in bouts_df.columns:
        return

    # Build mapping: (old_bout_id, chunk_start_frame) → new_bout_id
    # We detect chunk boundaries from frame_idx gaps in per-frame CSV
    pf_df = pd.read_csv(pf_path) if pf_path.exists() else None

    # Simpler approach: compute per-chunk bout counts from chunk dirs
    # and build offset mapping
    offset = 0
    chunk_bout_counts = []
    for cdir in chunk_dirs:
        bp = cdir / "contacts" / "contact_bouts.csv"
        if not bp.exists():
            bp = cdir / "contact_bouts.csv"
        if bp.exists():
            chunk_bouts = pd.read_csv(bp)
            n = len(chunk_bouts) if not chunk_bouts.empty else 0
        else:
            n = 0
        chunk_bout_counts.append((offset, n))
        offset += n

    # Renumber bouts_df sequentially
    bouts_df["bout_id"] = range(len(bouts_df))
    bouts_df.to_csv(bouts_path, index=False)

    # Renumber bout_id in per-frame CSV if it exists
    if pf_df is not None and not pf_df.empty and "bout_id" in pf_df.columns:
        # Rebuild per-frame bout_ids from bouts_df using frame_idx ranges
        if "start_frame" in bouts_df.columns and "end_frame" in bouts_df.columns:
            pf_df["bout_id"] = -1
            for _, bout in bouts_df.iterrows():
                mask = (
                    (pf_df["frame_idx"] >= bout["start_frame"])
                    & (pf_df["frame_idx"] <= bout["end_frame"])
                )
                pf_df.loc[mask, "bout_id"] = int(bout["bout_id"])
            pf_df.to_csv(pf_path, index=False)

    logger.info("Renumbered bout_ids: %d bouts across %d chunks", len(bouts_df), len(chunk_dirs))


def main():
    parser = argparse.ArgumentParser(
        description="Merge parallel chunk outputs into a single run.",
    )
    parser.add_argument(
        "chunk_dirs", nargs="+",
        help="Chunk output directories (e.g., outputs/runs/*_chunk*/).",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output directory. Default: outputs/runs/<timestamp>_merged/",
    )
    args = parser.parse_args()

    chunk_dirs = find_chunk_dirs(args.chunk_dirs)
    if not chunk_dirs:
        logger.error("No valid chunk directories found")
        sys.exit(1)

    logger.info("Found %d chunk directories:", len(chunk_dirs))
    for d in chunk_dirs:
        logger.info("  %s", d.name)

    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = Path("outputs/runs") / f"{timestamp}_merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s", output_dir)

    # Copy config from first chunk
    first_config = chunk_dirs[0] / "config_used.yaml"
    if first_config.exists():
        shutil.copy2(first_config, output_dir / "config_used.yaml")

    # Merge overlay videos
    merge_overlay_videos(chunk_dirs, output_dir)

    # Merge contact CSVs with bout_id renumbering to avoid collisions
    contacts_dir = output_dir / "contacts"
    contacts_dir.mkdir(parents=True, exist_ok=True)
    merge_csvs(chunk_dirs, "contacts_per_frame.csv", contacts_dir)
    merge_csvs(chunk_dirs, "contact_bouts.csv", contacts_dir)
    _renumber_bout_ids(chunk_dirs, contacts_dir)

    # Merge session summaries
    merge_session_summaries(chunk_dirs, output_dir)

    # Generate merged report PDF
    generate_merged_report(contacts_dir)

    # Post-process contacts into clean events
    try:
        from scripts.postprocess_contacts_simple import run_postprocess
        # Read FPS from merged session_summary.json
        _merged_summary = contacts_dir / "session_summary.json"
        _merge_fps = 30.0
        if _merged_summary.exists():
            with _merged_summary.open() as _f:
                _merge_fps = json.load(_f).get("metadata", {}).get("fps", 30.0)
        run_postprocess(contacts_dir, fps=_merge_fps)
    except Exception as e:
        logger.warning("Contact post-processing failed: %s", e)

    # Merge logs
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "run.log").open("w", encoding="utf-8") as out_log:
        for cdir in chunk_dirs:
            log_path = cdir / "logs" / "run.log"
            if log_path.exists():
                out_log.write(f"\n{'='*60}\n")
                out_log.write(f"=== Chunk: {cdir.name} ===\n")
                out_log.write(f"{'='*60}\n")
                out_log.write(log_path.read_text(encoding="utf-8"))

    logger.info("Merge complete → %s", output_dir)

    # Print final video location for easy copy-paste
    merged_videos = list((output_dir / "overlays").glob("*_merged.*")) if (output_dir / "overlays").exists() else []
    if merged_videos:
        print(f"\n{'='*60}")
        print(f"  MERGED VIDEO: {merged_videos[0]}")
        print(f"  OUTPUT DIR:   {output_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
