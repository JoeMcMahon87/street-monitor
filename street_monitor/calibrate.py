from __future__ import annotations

import sys

import cv2
import numpy as np

from .capture import VideoCapture
from .config import Config, save_config


def run_calibration(config: Config) -> None:
    print("=== Street Monitor Calibration ===")
    print("Opening camera...")

    clicks: list[tuple[int, int]] = []
    done = False

    def on_mouse(event, x, y, flags, param):
        nonlocal done
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4 and not done:
            clicks.append((x, y))
            print(f"  Point {len(clicks)}: ({x}, {y})")
            if len(clicks) == 4:
                done = True

    with VideoCapture(config.camera) as cap:
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Calibration", on_mouse)

        _LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"]
        _COLORS = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)
        ]

        print("\nClick the four corners of your known road rectangle in order:")
        print("  1. top-left   2. top-right   3. bottom-right   4. bottom-left")
        print("Press R to reset clicks, S to save, Q to quit without saving.\n")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: cannot read frame.", file=sys.stderr)
                break

            display = frame.copy()
            _draw_instructions(display, clicks, _LABELS)
            _draw_clicks(display, clicks, _LABELS, _COLORS)

            if len(clicks) == 4:
                _draw_quad(display, clicks)

            cv2.imshow("Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r') or key == ord('R'):
                clicks.clear()
                done = False
                print("Clicks reset.")
            elif key == ord('s') or key == ord('S'):
                if len(clicks) != 4:
                    print("Need exactly 4 clicks before saving.")
                    continue
                _prompt_and_save(config, clicks)
                break
            elif key == ord('q') or key == ord('Q'):
                print("Calibration cancelled.")
                break

    cv2.destroyAllWindows()


def _draw_instructions(frame: np.ndarray, clicks: list, labels: list) -> None:
    next_idx = len(clicks)
    if next_idx < 4:
        msg = f"Click point {next_idx + 1}: {labels[next_idx]}"
    else:
        msg = "Press S to save, R to reset"
    cv2.putText(frame, msg, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_clicks(frame: np.ndarray, clicks: list,
                 labels: list, colors: list) -> None:
    for i, (x, y) in enumerate(clicks):
        cv2.circle(frame, (x, y), 6, colors[i], -1)
        cv2.putText(frame, labels[i], (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[i], 1, cv2.LINE_AA)


def _draw_quad(frame: np.ndarray, clicks: list) -> None:
    pts = np.array(clicks, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 200, 200), thickness=2)


def _prompt_and_save(config: Config, clicks: list[tuple[int, int]]) -> None:
    print("\nEnter the real-world dimensions of the rectangle you clicked:")
    try:
        width_m = float(input("  Width (metres, left edge to right edge): ").strip())
        length_m = float(input("  Length (metres, top edge to bottom edge): ").strip())
    except (ValueError, EOFError):
        print("Invalid input. Calibration not saved.")
        return

    image_points = [list(p) for p in clicks]
    world_points = [
        [0.0, 0.0],
        [width_m, 0.0],
        [width_m, length_m],
        [0.0, length_m],
    ]

    config.calibration.image_points = image_points
    config.calibration.world_points = world_points

    path = "config.yaml"
    save_config(config, path)
    print(f"\nCalibration saved to {path}")
    print(f"  image_points: {image_points}")
    print(f"  world_points: {world_points}")
