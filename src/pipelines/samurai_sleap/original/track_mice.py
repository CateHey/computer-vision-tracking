"""
SAMURAI - Mouse Tracking Script (VS Code / Local)
Adapted from SAMURAI Colab notebook for local execution.

Usage:
    python track_mice.py --video path/to/video.mp4
    python track_mice.py --video path/to/video.mp4 --device cpu
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths — adjust if your repo clone is in a different location
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR / "samurai_repo"
SAM2_DIR = REPO_DIR / "sam2"
CHECKPOINT = REPO_DIR / "checkpoints" / "sam2.1_hiera_large.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Add SAMURAI and SAM2 to Python path
sys.path.insert(0, str(SAM2_DIR))
sys.path.insert(0, str(REPO_DIR))


def extract_frames(video_path: str, frames_dir: str):
    """Extract all frames from video as JPEGs."""
    os.makedirs(frames_dir, exist_ok=True)
    # Clean previous frames
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Extracting {total} frames ({width}x{height} @ {fps:.1f} FPS)...")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()
    return fps, width, height, idx


def select_bboxes_interactive(video_path: str):
    """
    Show first frame and let user draw bounding boxes with the mouse.
    Press ENTER after each box. Press 'q' or ESC when done.
    Returns list of dicts with x, y, w, h.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read video: {video_path}")

    print("\n=== BOUNDING BOX SELECTION ===")
    print("Draw a rectangle around each mouse, then press ENTER.")
    print("Press 'q' or ESC when done selecting all mice.\n")

    bboxes = []
    clone = frame.copy()
    window_name = "Select mice - Draw box + ENTER | 'q' to finish"

    while True:
        # Show current boxes drawn so far
        display = clone.copy()
        colors = [(0, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255),
                  (255, 165, 0), (128, 0, 128)]
        for i, bb in enumerate(bboxes):
            c = colors[i % len(colors)]
            cv2.rectangle(display, (bb["x"], bb["y"]),
                          (bb["x"] + bb["w"], bb["y"] + bb["h"]), c, 2)
            cv2.putText(display, f"Mouse {i+1}", (bb["x"], bb["y"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

        roi = cv2.selectROI(window_name, display, fromCenter=False, showCrosshair=True)
        x, y, w, h = roi

        if w == 0 or h == 0:
            # User pressed ESC or q
            break

        bboxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        print(f"  Mouse {len(bboxes)}: x={x}, y={y}, w={w}, h={h}")

    cv2.destroyAllWindows()

    if not bboxes:
        raise RuntimeError("No bounding boxes selected. Exiting.")

    print(f"\n{len(bboxes)} mouse(es) selected.\n")
    return bboxes


def run_tracking(frames_dir, bboxes, device, checkpoint, config):
    """Run SAMURAI tracking over extracted frames."""
    from sam2.build_sam import build_sam2_video_predictor

    print(f"Loading SAMURAI model on {device}...")

    # SAM2 expects to find configs relative to its own directory
    original_dir = os.getcwd()
    os.chdir(str(SAM2_DIR))

    predictor = build_sam2_video_predictor(config, str(checkpoint), device=device)

    use_amp = device == "cuda"
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.inference_mode()

    with torch.inference_mode(), ctx:
        state = predictor.init_state(frames_dir)

        # Register each bounding box
        for i, bb in enumerate(bboxes):
            obj_id = i + 1
            box = np.array([bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]],
                           dtype=np.float32)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, box=box)
            print(f"  Registered Mouse {obj_id}: bbox={box.tolist()}")

        # Propagate
        print("Tracking...")
        total_frames = len(os.listdir(frames_dir))
        video_segments = {}
        for frame_idx, obj_ids, mask_logits in tqdm(
            predictor.propagate_in_video(state), total=total_frames, desc="Tracking"
        ):
            video_segments[frame_idx] = {
                obj_id: (mask_logits[i] > 0.0).cpu().numpy().squeeze()
                for i, obj_id in enumerate(obj_ids)
            }

    os.chdir(original_dir)
    return video_segments


def render_output(frames_dir, video_segments, output_path, fps, width, height, num_objects):
    """Render tracked masks onto frames and write output video."""
    # Color palette (BGR)
    palette = [
        (80, 80, 255),    # red
        (255, 220, 80),   # cyan
        (80, 255, 80),    # green
        (255, 80, 255),   # magenta
        (0, 165, 255),    # orange
        (128, 0, 128),    # purple
    ]

    print(f"Rendering output to {output_path}...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_files = sorted(os.listdir(frames_dir))
    for frame_idx, fname in enumerate(tqdm(frame_files, desc="Rendering")):
        img = cv2.imread(os.path.join(frames_dir, fname))
        overlay = img.copy()

        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                color_bgr = palette[(obj_id - 1) % len(palette)]
                colored = np.zeros_like(img)
                colored[mask] = color_bgr
                overlay = cv2.addWeighted(overlay, 1.0, colored, 0.45, 0)

                contours, _ = cv2.findContours(
                    (mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, color_bgr, 2)

                ys, xs = np.where(mask)
                if len(xs) > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    cv2.putText(overlay, f"Mouse {obj_id}", (cx - 40, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)

        writer.write(overlay)

    writer.release()
    print(f"Done! Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SAMURAI Mouse Tracker (Local)")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video file (mp4, avi, mov)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: <video>_tracked.mp4)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda' or 'cpu' (auto-detected if omitted)")
    parser.add_argument("--frames-dir", type=str, default=None,
                        help="Temp directory for extracted frames")
    # Manual bbox mode (skip interactive selection)
    parser.add_argument("--bbox", type=str, action="append", default=None,
                        help="Manual bbox as 'x,y,w,h'. Use multiple --bbox for multiple mice.")
    args = parser.parse_args()

    # Validate video
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: No GPU detected. Running on CPU will be very slow.")
            print("         For better performance, install CUDA-enabled PyTorch.\n")

    # Output path
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = video_path.parent / f"{video_path.stem}_tracked.mp4"

    # Frames dir
    frames_dir = args.frames_dir or str(SCRIPT_DIR / "frames_tmp")

    # Check checkpoint exists
    if not CHECKPOINT.exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT}")
        print("Run setup.bat first, or download manually:")
        print("  pip install huggingface_hub")
        print("  python -c \"from huggingface_hub import hf_hub_download; "
              "hf_hub_download(repo_id='facebook/sam2.1-hiera-large', "
              "filename='sam2.1_hiera_large.pt', local_dir='samurai_repo/checkpoints')\"")
        sys.exit(1)

    # Get bounding boxes
    if args.bbox:
        bboxes = []
        for b in args.bbox:
            parts = [int(v) for v in b.split(",")]
            if len(parts) != 4:
                print(f"ERROR: Invalid bbox format '{b}'. Expected 'x,y,w,h'")
                sys.exit(1)
            bboxes.append({"x": parts[0], "y": parts[1], "w": parts[2], "h": parts[3]})
    else:
        bboxes = select_bboxes_interactive(str(video_path))

    # Extract frames
    fps, width, height, num_frames = extract_frames(str(video_path), frames_dir)

    # Run tracking
    video_segments = run_tracking(frames_dir, bboxes, device, str(CHECKPOINT), CONFIG)

    # Render output
    render_output(frames_dir, video_segments, str(output_path), fps, width, height, len(bboxes))

    print(f"\nTracked {len(bboxes)} mouse(es) across {num_frames} frames.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
