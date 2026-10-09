"""Manual frozen-pair capture using the unchanged proxy.viewer.FrameParser."""

import argparse
import logging
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from proxy.viewer import FrameParser
from .dataset import CapturedFrame, decode_jpeg, save_pair, validate_pair
from .orientation import (
    ROTATIONS, OrientationSettings, add_orientation_arguments, orientation_from_args,
    rotate_image, save_orientation,
)


LOG = logging.getLogger(__name__)
TITLE = "VIDEX | 0 LEFT / 1 RIGHT (physical mapping unverified)"


def receive_frames(port, baudrate, stop, latest, lock):
    import serial

    while not stop.is_set():
        try:
            with serial.Serial(port, baudrate=baudrate, timeout=0.1) as connection:
                parser = FrameParser()
                LOG.info("Connected: %s", port)
                while not stop.is_set():
                    data = connection.read(min(max(connection.in_waiting, 1), 65536))
                    received_at, received_ns = time.time(), time.monotonic_ns()
                    for frame in parser.feed(data):
                        try:
                            captured = CapturedFrame.from_wire(frame, received_at, received_ns)
                        except (ValueError, cv2.error) as error:
                            LOG.warning("Rejected camera %s frame %s: %s",
                                        frame.camera_id, frame.frame_id, error)
                            continue
                        with lock:
                            latest[captured.camera_id] = captured
        except (serial.SerialException, OSError) as error:
            with lock:
                latest.clear()  # Never combine frames from different connections.
            LOG.warning("Serial unavailable: %s; reconnecting in 1s", error)
            stop.wait(1)


def freeze_pair(latest, max_age, now_ns=None, orientation=None):
    left, right = latest.get(0), latest.get(1)
    validate_pair(left, right, orientation)
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    if any((now_ns - frame.received_monotonic_ns) / 1e9 > max_age for frame in (left, right)):
        raise ValueError("Frames are stale; wait for fresh LEFT and RIGHT images")
    return left, right


def preview_canvas(frames, message, orientation=None):
    orientation = OrientationSettings() if orientation is None else orientation
    canvas = np.zeros((570, 1280, 3), np.uint8)
    for camera_id, side in ((0, "LEFT"), (1, "RIGHT")):
        x = 640 * camera_id
        frame = frames.get(camera_id)
        label = f"{side}: waiting"
        if frame is not None:
            rotation = orientation.left_rotation if camera_id == 0 else orientation.right_rotation
            image = rotate_image(decode_jpeg(frame.jpeg, frame.width, frame.height), rotation)
            height, width = image.shape[:2]
            scale = min(640 / width, 480 / height)
            resized = cv2.resize(image, (max(1, round(width * scale)),
                                        max(1, round(height * scale))))
            h, w = resized.shape[:2]
            left, top = x + (640 - w) // 2, 40 + (480 - h) // 2
            canvas[top:top + h, left:left + w] = resized
            label = f"{side}: rot={rotation} raw={frame.width}x{frame.height} -> {width}x{height}"
        cv2.putText(canvas, label, (x + 10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Sync UNVERIFIED | keep board still | physical L/R unverified",
                (10, 537), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    cv2.putText(canvas, message[:140], (10, 562), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    return canvas


def preview(frames, message, orientation=None):
    cv2.imshow(TITLE, preview_canvas(frames, message, orientation))


def rotation_choices_preview(left, right):
    """Two camera rows, four labeled rotations each; input arrays remain raw."""
    canvas = np.zeros((600, 1280, 3), np.uint8)
    for row, (side, image) in enumerate((("LEFT", left), ("RIGHT", right))):
        for col, rotation in enumerate(ROTATIONS):
            rotated = rotate_image(image, rotation)
            height, width = rotated.shape[:2]
            scale = min(310 / width, 250 / height)
            resized = cv2.resize(rotated, (max(1, round(width * scale)), max(1, round(height * scale))))
            h, w = resized.shape[:2]
            x, y = col * 320 + (320 - w) // 2, row * 300 + 40 + (250 - h) // 2
            canvas[y:y + h, x:x + w] = resized
            cv2.putText(canvas, f"{side} {rotation}: {width}x{height}", (col * 320 + 6, row * 300 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    return canvas


def run_orientation_preview(args):
    frames = {0: CapturedFrame.from_file(args.left, 0), 1: CapturedFrame.from_file(args.right, 1)}
    settings = orientation_from_args(args) or OrientationSettings()
    if args.comparison_output is not None:
        output = args.comparison_output
        if output.suffix.lower() != ".png" or output.exists():
            raise ValueError("Comparison output must be a new .png file")
        output.parent.mkdir(parents=True, exist_ok=True)
        image = rotation_choices_preview(*(decode_jpeg(frames[i].jpeg) for i in (0, 1)))
        if not cv2.imwrite(str(output), image):
            raise OSError("Could not write rotation comparison PNG")
        print(f"Rotation choices preview written: {output}; actual orientation is unverified")
    if args.no_gui:
        return 0
    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    message = "A: cycle LEFT | D: cycle RIGHT | S: save settings | Q/ESC: cancel"
    try:
        while True:
            preview(frames, message, settings)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                return 0
            if key in (ord("a"), ord("A")):
                settings = settings.cycle(0)
            elif key in (ord("d"), ord("D")):
                settings = settings.cycle(1)
            elif key in (ord("s"), ord("S")):
                try:
                    validate_pair(frames[0], frames[1], settings)
                    save_orientation(args.output, settings)
                    print(f"Orientation profile written: {args.output}")
                    print(f"LEFT={settings.left_rotation}, RIGHT={settings.right_rotation}; "
                          "physical mapping and exposure sync unverified")
                    return 0
                except (ValueError, OSError) as error:
                    message = str(error)
                    LOG.warning("Settings not saved: %s", error)
    finally:
        cv2.destroyAllWindows()


def run_live(args):
    import serial  # Fail in the main thread if the dependency is missing.

    stop, lock, latest = threading.Event(), threading.Lock(), {}
    worker = threading.Thread(target=receive_frames,
                              args=(args.port, args.baudrate, stop, latest, lock), daemon=True)
    frozen, last_saved = None, None
    orientation = orientation_from_args(args) or OrientationSettings()
    message = "A/D: rotate LEFT/RIGHT | F: freeze | S: save frozen | R: resume | Q/ESC: quit"
    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    try:
        worker.start()
        while True:
            with lock:
                snapshot = latest.copy()
            displayed = snapshot if frozen is None else {0: frozen[0], 1: frozen[1]}
            preview(displayed, message, orientation)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
            try:
                if key in (ord("a"), ord("A"), ord("d"), ord("D")):
                    orientation = orientation.cycle(0 if key in (ord("a"), ord("A")) else 1)
                    message = "Rotation changed; inspect both images. Use a new session if settings change after saving."
                elif key in (ord("f"), ord("F")):
                    frozen = freeze_pair(snapshot, args.max_age, orientation=orientation)
                    delta = abs(frozen[0].received_monotonic_ns - frozen[1].received_monotonic_ns) / 1e6
                    message = f"FROZEN | PC arrival delta={delta:.1f}ms (not exposure sync) | inspect both, then S"
                elif key in (ord("r"), ord("R")):
                    frozen = None
                    message = "LIVE | F: freeze | S: save frozen | R: resume | Q: quit"
                elif key in (ord("s"), ord("S")):
                    if frozen is None:
                        raise ValueError("Press F and inspect both frozen images before saving")
                    if frozen == last_saved:
                        raise ValueError("This exact frozen pair was already saved; resume and capture a new pose")
                    path = save_pair(args.output, *frozen, orientation=orientation)
                    last_saved = frozen
                    LOG.info("Saved manually selected pair (sync unverified): %s", path)
                    message = "Saved frozen pair | R: resume and change board pose"
            except (ValueError, OSError, cv2.error) as error:
                LOG.warning("Capture refused: %s", error)
                message = str(error)
    finally:
        stop.set()
        if worker.is_alive():
            worker.join(timeout=2)
        cv2.destroyAllWindows()
    return 0


def import_pair(args):
    left = CapturedFrame.from_file(args.left, 0)
    right = CapturedFrame.from_file(args.right, 1)
    orientation = orientation_from_args(args) or OrientationSettings()
    validate_pair(left, right, orientation)
    if not args.save_without_preview:
        cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
        try:
            while True:
                preview({0: left, 1: right}, "Offline selected pair | inspect both | S: save | Q/ESC: cancel", orientation)
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("s"), ord("S")):
                    break
                if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    return 0
        finally:
            cv2.destroyAllWindows()
    path = save_pair(args.output, left, right, orientation)
    print(f"Saved manually selected pair (sync unverified): {path}")
    print("Imported files have unknown camera frame_id/timestamp_us; PC times are import times.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    live = sub.add_parser("live", help="Native USB preview, freeze, then save")
    live.add_argument("--port", required=True)
    live.add_argument("--baudrate", type=int, default=115200)
    live.add_argument("--max-age", type=float, default=2.0, help="Maximum PC frame age when freezing, seconds")
    live.add_argument("--output", type=Path, required=True)
    add_orientation_arguments(live)
    offline = sub.add_parser("import", help="Manually select two existing JPEGs")
    offline.add_argument("--left", type=Path, required=True)
    offline.add_argument("--right", type=Path, required=True)
    offline.add_argument("--output", type=Path, required=True)
    offline.add_argument("--save-without-preview", action="store_true",
                         help="Explicitly save the selected files without GUI review (for offline automation)")
    add_orientation_arguments(offline)
    directions = sub.add_parser("preview", help="Inspect each camera's four rotations and save a profile")
    directions.add_argument("--left", type=Path, required=True)
    directions.add_argument("--right", type=Path, required=True)
    directions.add_argument("--output", type=Path, help="New orientation JSON; required for interactive selection")
    directions.add_argument("--comparison-output", type=Path, help="Export an eight-tile rotation comparison PNG")
    directions.add_argument("--no-gui", action="store_true", help="Only export the comparison; do not choose settings")
    add_orientation_arguments(directions)
    args = parser.parse_args(argv)
    if args.command == "live" and (not np.isfinite(args.max_age) or args.max_age <= 0):
        parser.error("--max-age must be finite and positive")
    if args.command == "preview":
        if args.no_gui and args.comparison_output is None:
            parser.error("--no-gui requires --comparison-output")
        if not args.no_gui and args.output is None:
            parser.error("Interactive preview requires --output for the orientation JSON")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "preview":
            return run_orientation_preview(args)
        return run_live(args) if args.command == "live" else import_pair(args)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, cv2.error, ImportError) as error:
        LOG.error("Capture stopped: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
