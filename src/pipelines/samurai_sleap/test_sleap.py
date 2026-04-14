"""Quick diagnostic to test SLEAP full pipeline: model → confmaps → peaks → mask assignment."""

import sys
import torch
import numpy as np
import cv2
from src.pipelines.samurai_sleap.sleap_model import load_sleap_checkpoint, predict_keypoints
from src.pipelines.samurai_sleap.run import sleap_predict_peaks, assign_peaks_to_masks, KEYPOINT_NAMES

device = "cuda" if torch.cuda.is_available() else "cpu"
video_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/original_120s.avi"

print(f"Device: {device}")
model = load_sleap_checkpoint("models/sleap/sleap.ckpt", device=device)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print(f"Frame shape: {frame_rgb.shape}")

# --- Test 1: Raw model output ---
img = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
img = img.to(device)

with torch.inference_mode():
    confmaps, pafs = model(img)

cm_raw = confmaps.cpu().numpy().squeeze()
print(f"\nRaw confmaps — shape: {cm_raw.shape}")
print(f"  min: {cm_raw.min():.4f}  max: {cm_raw.max():.4f}")
for i in range(min(7, cm_raw.shape[0])):
    print(f"  Ch {i} ({KEYPOINT_NAMES[i]}): min={cm_raw[i].min():.4f} max={cm_raw[i].max():.4f}")

# --- Test 2: predict_keypoints (peak detection) ---
print("\n--- predict_keypoints test ---")
try:
    confmaps_np, peaks = predict_keypoints(
        model, frame_rgb, KEYPOINT_NAMES,
        min_confidence=0.3, device=device,
    )
    print(f"Peaks found: {len(peaks)}")
    for kpt_idx, x, y, conf in peaks:
        name = KEYPOINT_NAMES[kpt_idx] if kpt_idx < len(KEYPOINT_NAMES) else f"kpt_{kpt_idx}"
        print(f"  {name}: ({x:.0f}, {y:.0f}) conf={conf:.3f}")
except Exception as e:
    print(f"ERROR in predict_keypoints: {e}")
    import traceback
    traceback.print_exc()
    peaks = []

# --- Test 3: assign_peaks_to_masks (with fake masks from init bboxes) ---
print("\n--- assign_peaks_to_masks test (using bbox masks) ---")
if peaks:
    h, w = frame_rgb.shape[:2]
    # Create simple rectangular masks from the config bboxes
    bboxes = [
        (424, 701, 309, 147),  # Rat 1
        (105, 709, 383, 179),  # Rat 2
    ]
    masks = []
    centroids = []
    for bx, by, bw, bh in bboxes:
        mask = np.zeros((h, w), dtype=bool)
        mask[by:by+bh, bx:bx+bw] = True
        masks.append(mask)
        centroids.append((bx + bw/2, by + bh/2))

    slot_dets = assign_peaks_to_masks(peaks, KEYPOINT_NAMES, masks, centroids)

    for i, det in enumerate(slot_dets):
        if det is None:
            print(f"  Rat {i+1}: no keypoints assigned")
        else:
            print(f"  Rat {i+1}: {len(det.keypoints)} keypoints")
            for kp in det.keypoints:
                print(f"    {kp.name}: ({kp.x:.0f}, {kp.y:.0f}) conf={kp.conf:.3f}")

    # Draw on frame
    from src.common.visualization import draw_keypoints
    frame_out = frame_rgb.copy()
    dets_to_draw = [d for d in slot_dets if d is not None]
    if dets_to_draw:
        frame_out = draw_keypoints(frame_out, dets_to_draw, min_conf=0.3)
        # Also draw the bbox masks as outlines
        for i, (bx, by, bw, bh) in enumerate(bboxes):
            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            cv2.rectangle(frame_out, (bx, by), (bx+bw, by+bh), color, 2)
        cv2.imwrite("test_sleap_keypoints.jpg", cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR))
        print("\nSaved test_sleap_keypoints.jpg")
    else:
        print("\nNo detections to draw!")
else:
    print("No peaks to assign!")

# Save confmap heatmaps
for i in range(min(7, cm_raw.shape[0])):
    cmap = cm_raw[i]
    cmap_vis = (cmap / max(cmap.max(), 1e-6) * 255).astype(np.uint8)
    cmap_vis = cv2.applyColorMap(cmap_vis, cv2.COLORMAP_JET)
    cv2.imwrite(f"confmap_ch{i}.jpg", cmap_vis)
print("Saved confmap_ch0..ch6.jpg")
