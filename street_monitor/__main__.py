from __future__ import annotations

import os
# Must be set before cv2 (and its bundled Qt) is imported.
# Force X11/XWayland so Qt doesn't warn about the Wayland session type.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
# Point Qt's font database at system fonts; opencv-python ships without them.
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

import argparse
import shutil
import sys

# Ensure stdout is line-buffered even when piped (e.g. | tee debug.log).
# Without this, Python switches to block-buffering and debug output appears
# in large delayed chunks rather than as each event occurs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import cv2

from .calibrate import run_calibration
from .capture import VideoCapture
from .config import load_config
from .pipeline import Pipeline
from .recorder import Recorder
from .renderer import Renderer
from .utils import FPSCounter


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="street-monitor",
        description="Vehicle speed detection using YOLO and a webcam.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=["run", "calibrate", "replay"],
        help="Operating mode (default: run)",
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config file (default: config.yaml)")
    parser.add_argument("--input", metavar="VIDEO",
                        help="Video file path for replay mode")
    parser.add_argument("--headless", action="store_true",
                        help="Run without a display window (recording only)")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="Stop after N frames (useful for testing)")
    parser.add_argument("--debug", action="store_true",
                        help="Print track lifecycle and traversal state to stdout")
    args = parser.parse_args()

    # Bootstrap config.yaml from example if missing
    if not os.path.exists(args.config):
        example = os.path.join(os.path.dirname(__file__), "..", "config.example.yaml")
        if os.path.exists(example):
            shutil.copy(example, args.config)
            print(f"Created {args.config} from config.example.yaml")
        else:
            print(f"Warning: {args.config} not found. Using defaults.")

    config = load_config(args.config)

    if args.mode == "calibrate":
        run_calibration(config)
        return

    # For replay mode, override camera device_id with the provided file
    if args.mode == "replay":
        if not args.input:
            print("Error: --input is required for replay mode.", file=sys.stderr)
            sys.exit(1)
        config.camera.device_id = args.input

    run_pipeline(config, args)


def run_pipeline(config, args) -> None:
    print("Starting Street Monitor...")
    print(f"  Camera/source: {config.camera.device_id}")
    print(f"  Model: {config.detection.model}  device: {config.detection.device}")
    print(f"  Entry line X={config.lines.entry_x}  Exit line X={config.lines.exit_x}")

    with VideoCapture(config.camera) as cap:
        fps = cap.get_fps()
        print(f"  FPS: {fps:.1f}")

        pipeline = Pipeline(config, fps, debug=args.debug)
        renderer = Renderer(config.display, config.lines)

        with Recorder(config.recording) as recorder:
            fps_counter = FPSCounter(window=30)
            frame_number = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    print("End of stream.")
                    break

                result = pipeline.process_frame(frame, frame_number)

                for completed in result.completed:
                    print(
                        f"  [{completed.timestamp.strftime('%H:%M:%S')}] "
                        f"{completed.class_name} #{completed.track_id} "
                        f"({completed.direction}): "
                        f"avg {completed.avg_speed_kmh} "
                        f"min {completed.min_speed_kmh} "
                        f"max {completed.max_speed_kmh} km/h"
                    )
                    recorder.record_detection(completed, frame)

                fps_counter.tick()
                annotated = renderer.draw(frame, result, fps=fps_counter.get_fps())

                if not args.headless:
                    cv2.imshow("Street Monitor", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:   # q or Esc
                        print("Quit.")
                        break

                frame_number += 1
                if args.limit and frame_number >= args.limit:
                    print(f"Reached frame limit ({args.limit}).")
                    break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
