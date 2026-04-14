"""
Interactive tool to select initial bounding boxes from a video's first frame.

Opens the first frame, lets you draw boxes around each rat, then prints
the init_bboxes config you can paste into your YAML config.

Usage:
    python -m src.pipelines.samurai_sleap.get_init_bboxes --video data/clips/output-10s.mp4
"""

import argparse
import cv2
import sys


def main():
    parser = argparse.ArgumentParser(description="Select init bounding boxes from first frame")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"ERROR: Cannot read video: {args.video}")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")
    print()
    print("Draw a rectangle around each rat, then press ENTER.")
    print("Press 'q' or ESC when done selecting all rats.")
    print()

    bboxes = []
    clone = frame.copy()
    colors = [(0, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255)]
    window_name = "Select rats - Draw box + ENTER | ESC to finish"

    while True:
        display = clone.copy()
        for i, bb in enumerate(bboxes):
            c = colors[i % len(colors)]
            cv2.rectangle(display, (bb[0], bb[1]),
                          (bb[0] + bb[2], bb[1] + bb[3]), c, 2)
            cv2.putText(display, f"Rat {i+1}", (bb[0], bb[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

        roi = cv2.selectROI(window_name, display, fromCenter=False, showCrosshair=True)
        x, y, w_box, h_box = roi

        if w_box == 0 or h_box == 0:
            break

        bboxes.append([int(x), int(y), int(w_box), int(h_box)])
        print(f"  Rat {len(bboxes)}: x={x}, y={y}, w={w_box}, h={h_box}")

    cv2.destroyAllWindows()

    if not bboxes:
        print("No boxes selected.")
        sys.exit(1)

    # Print YAML snippet
    print()
    print("=" * 50)
    print("Paste this into your YAML config:")
    print("=" * 50)
    print()
    print("init_bboxes:")
    for i, bb in enumerate(bboxes):
        print(f"  - [{bb[0]}, {bb[1]}, {bb[2]}, {bb[3]}]   # Rat {i+1}")
    print()


if __name__ == "__main__":
    main()
