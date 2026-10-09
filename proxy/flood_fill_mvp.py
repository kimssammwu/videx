#!/usr/bin/env python3
"""VIDEX USB/demo viewer for the reusable flood_fill detector."""

import argparse
import json
import logging
import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np

if __package__:
    from .flood_fill import Config, Detector, analyze, extract_outline
    from .outline import receive_frames
else:
    from flood_fill import Config, Detector, analyze, extract_outline
    from outline import receive_frames


def render(detector: Detector, now: float, source: str) -> np.ndarray:
    canvas = np.full((740, 1280, 3), 22, np.uint8)
    colors = {"WARNING": (40, 40, 255), "MONITORING": (80, 220, 80),
              "UNKNOWN": (0, 210, 255)}
    color = colors[detector.state]
    def text(value, x, y, size=.65, tint=(230, 230, 230)):
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size, tint, 1, cv2.LINE_AA)
    text("VIDEX | " + detector.state, 20, 37, 1.0, color)
    text(detector.reason, 20, 68, .65, color)
    text(source + " | camera 0 LEFT | Q/ESC quit | S snapshot", 20, 96, .60)
    r = detector.result
    if r is not None:
        overlay = r["image"].copy()
        if detector.state != "UNKNOWN":
            for mask, tint in ((r["filled"], (0, 220, 0)), (r["unfilled"], (0, 0, 255))):
                overlay[mask] = (overlay[mask] * .45 + np.array(tint) * .55).astype(np.uint8)
        else:
            # Do not present a stale/invalid green area as a current result.
            overlay = cv2.cvtColor(cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        cv2.polylines(overlay, [r["polygon"]], True, (255, 70, 0), 2)
        contours, _ = cv2.findContours(r["center"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 1)
        if r["seed"] is not None:
            cv2.drawMarker(overlay, r["seed"], (255, 255, 255), cv2.MARKER_CROSS, 12, 2)
        panels = [r["image"], cv2.cvtColor(r["edges"], cv2.COLOR_GRAY2BGR), overlay]
        for i, (panel, title) in enumerate(zip(panels, ("CAMERA (90 CW)", "CANNY (existing extract_outline)", "FLOOD FILL / ROI"))):
            x = i * 426
            text(title, x + 12, 125, .56)
            h, w = panel.shape[:2]
            scale = min(410 / w, 460 / h)
            resized = cv2.resize(panel, (max(1, round(w * scale)), max(1, round(h * scale))))
            ph, pw = resized.shape[:2]
            left, top = x + (426 - pw) // 2, 138 + (460 - ph) // 2
            canvas[top:top + ph, left:left + pw] = resized
        age = max(0., now - detector.last_received)
        prefix = "LAST / INVALID " if detector.state == "UNKNOWN" else ""
        text(f"{prefix}Unfilled: {r['total']:.1%} / {detector.config.total_threshold:.0%}    Central: {r['central']:.1%} / {detector.config.center_threshold:.0%}", 20, 628, .75, color)
        text(f"NEW frame {detector.last_id} | age {age:.2f}s | confirm {min(detector.hits, detector.config.confirm)}/{detector.config.confirm} | clear {min(detector.clears, detector.config.confirm)}/{detector.config.confirm} | barriers {r['edge_density']:.1%}", 20, 658, .62)
    text("BLUE: path ROI   GREEN: reachable   RED: unfilled   CYAN: center   WHITE +: seed", 20, 692, .60)
    text("DEMO HEURISTIC: floor patterns / missing edges can cause errors. No depth or distance.", 20, 721, .59, (0, 200, 255))
    return canvas


def synthetic(index: int) -> np.ndarray:
    """Repeat 3s clear / 4s central obstacle / 3s clear at 10 FPS."""
    image = np.full((320, 240, 3), 145, np.uint8)
    if 30 <= index % 100 < 70:
        cv2.rectangle(image, (82, 165), (158, 260), (25, 25, 25), -1)
    return image


def main() -> int:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--demo", action="store_true", help="Synthetic scene; no USB")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=0, help="0 = until Q/Ctrl+C")
    parser.add_argument("--output-dir", type=Path, help="Save state changes, last.png and report.json")
    parser.add_argument("--low", type=int, default=defaults.low)
    parser.add_argument("--high", type=int, default=defaults.high)
    parser.add_argument("--kernel", type=int, default=defaults.kernel)
    parser.add_argument("--total-threshold", type=float, default=defaults.total_threshold)
    parser.add_argument("--center-threshold", type=float, default=defaults.center_threshold)
    parser.add_argument("--confirm", type=int, default=defaults.confirm)
    parser.add_argument("--stale", type=float, default=defaults.stale)
    args = parser.parse_args()
    try:
        config = Config(low=args.low, high=args.high, kernel=args.kernel,
                        total_threshold=args.total_threshold, center_threshold=args.center_threshold,
                        confirm=args.confirm, stale=args.stale)
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("seconds must be finite and nonnegative")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    detector = Detector(config)
    latest, lock, stop = {}, threading.Lock(), threading.Event()
    worker = None
    if not args.demo:
        import serial
        # Reuse original receiver/parser/decoder, including local CAM2 fixes.
        worker = threading.Thread(target=receive_frames,
                                  args=(args.port, args.baudrate, stop, latest, lock, serial, cv2, np),
                                  daemon=True)
    title = "VIDEX FLOOD FILL | LEFT | Q: quit"
    source = "SYNTHETIC DEMO" if args.demo else "USB " + args.port
    started = time.monotonic()
    last_demo, previous_state = -1, None
    transitions = []
    canvas = None
    exit_code = 0
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not args.headless:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 1280, 740)
        if worker:
            worker.start()
        while args.seconds == 0 or time.monotonic() - started < args.seconds:
            now = time.monotonic()
            if args.demo:
                frame_id = int((now - started) * 10)
                if frame_id != last_demo:
                    detector.update(synthetic(frame_id), frame_id, now, now)
                    last_demo = frame_id
            else:
                with lock:
                    item = latest.get(0)  # RIGHT never contributes to decisions.
                if item is not None:
                    raw, frame_id, received_at = item
                    if frame_id != detector.last_id:
                        # Original outline.py LEFT mount correction, unchanged.
                        image = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
                        detector.update(image, frame_id, received_at, now)
            detector.tick(now)
            canvas = render(detector, now, source)
            if detector.state != previous_state:
                r = detector.result
                event = dict(seconds=round(now - started, 3), state=detector.state,
                             reason=detector.reason, frame_id=detector.last_id,
                             total=r["total"] if r else None, central=r["central"] if r else None)
                transitions.append(event)
                logging.info("%s", json.dumps(event))
                previous_state = detector.state
                if args.output_dir:
                    save_image(args.output_dir / f"{len(transitions):03d}_{detector.state.lower()}.png", canvas)
            if args.headless:
                stop.wait(.015)
            else:
                cv2.imshow(title, canvas)
                key = cv2.waitKey(15) & 0xFF
                if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("s"), ord("S")):
                    path = (args.output_dir or Path.cwd()) / "flood_snapshot.png"
                    save_image(path, canvas)
                    logging.info("Saved %s", path)
    except KeyboardInterrupt:
        pass
    except (cv2.error, OSError) as exc:
        logging.error("Run failed: %s", exc)
        exit_code = 1
    finally:
        stop.set()
        if worker and worker.is_alive():
            worker.join(timeout=2)
        if not args.headless:
            cv2.destroyAllWindows()
        report = dict(source=source, processed_new_frames=detector.processed,
                      state=detector.state, reason=detector.reason, transitions=transitions)
        if args.output_dir:
            if canvas is not None:
                save_image(args.output_dir / "last.png", canvas)
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    return exit_code


def save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save {path}")


if __name__ == "__main__":
    raise SystemExit(main())
