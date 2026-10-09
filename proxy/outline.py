#!/usr/bin/env python3
"""Extract outlines from CAM2 USB frames or a saved image using Canny edges."""

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import struct
import threading
import time


HEADER = struct.Struct("<IB3xIIHHQ")
MAGIC = b"CAM2"
MAX_JPEG_SIZE = 1024 * 1024
FRAME_TIMEOUT = 6.0


@dataclass(frozen=True)
class Frame:
    camera_id: int
    frame_id: int
    width: int
    height: int
    timestamp_us: int
    jpeg: bytes


class FrameParser:
    """Incremental parser; USB reads need not match frame boundaries."""

    def __init__(self):
        self.buffer = bytearray()
        self.started = None

    def feed(self, data, now=None):
        now = time.monotonic() if now is None else now
        self.buffer.extend(data)
        frames = []
        while True:
            offset = self.buffer.find(MAGIC)
            if offset < 0:
                # Retain a possible partial magic at the end of a read.
                self.buffer[:] = self.buffer[-3:]
                self.started = None
                break
            if offset:
                del self.buffer[:offset]
                self.started = None
            if self.started is None:
                self.started = now
            if now - self.started >= FRAME_TIMEOUT:
                del self.buffer[0]
                self.started = None
                continue
            if len(self.buffer) < HEADER.size:
                break
            _, camera_id, frame_id, size, width, height, timestamp = HEADER.unpack_from(self.buffer)
            if not (
                camera_id in (0, 1)
                and self.buffer[5:8] == b"\x00\x00\x00"
                and 4 <= size <= MAX_JPEG_SIZE
                and 0 < width <= 2000
                and 0 < height <= 2000
            ):
                del self.buffer[0]
                self.started = None
                continue
            end = HEADER.size + size
            if len(self.buffer) < end:
                break
            jpeg = bytes(self.buffer[HEADER.size:end])
            if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
                del self.buffer[0]
                self.started = None
                continue
            frames.append(Frame(camera_id, frame_id, width, height, timestamp, jpeg))
            del self.buffer[:end]
            self.started = None
        return frames


def receive_frames(port, baudrate, stop, latest, lock, serial, cv2, np):
    """Keep serial reads and JPEG decoding off the GUI thread."""
    while not stop.is_set():
        try:
            with serial.Serial(port, baudrate=baudrate, timeout=0.1) as connection:
                logging.info("USB connected: %s", port)
                parser = FrameParser()
                while not stop.is_set():
                    data = connection.read(min(max(connection.in_waiting, 1), 65536))
                    # Empty reads also let the parser expire incomplete frames.
                    for frame in parser.feed(data):
                        image = cv2.imdecode(np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if image is None or image.shape[:2] != (frame.height, frame.width):
                            logging.warning("Invalid JPEG or dimensions: camera %d frame %d",
                                            frame.camera_id, frame.frame_id)
                            continue
                        with lock:
                            latest[frame.camera_id] = (image, frame.frame_id, time.monotonic())
        except (serial.SerialException, OSError) as error:
            logging.warning("USB unavailable: %s; retrying in 1 second", error)
            stop.wait(1.0)



def extract_outline(image, cv2, low=50, high=150):
    """Return a white-on-black outline at the original image resolution."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, low, high)


def save_outline(path, outline, cv2):
    if not cv2.imwrite(str(path), outline):
        raise OSError(f"Could not write outline: {path}")



def main():
    args_parser = argparse.ArgumentParser(description=__doc__)
    args_parser.add_argument("--port", help="Native USB port, e.g. /dev/ttyACM0 or COM5")
    args_parser.add_argument("--list-ports", action="store_true", help="List ports and exit")
    args_parser.add_argument("--baudrate", type=int, default=115200,
                             help="CDC line setting (native USB speed is independent of baudrate)")
    args_parser.add_argument("--image", type=Path, help="Extract a saved image instead of USB frames")
    args_parser.add_argument("--output", type=Path, help="Output image path (requires --image)")
    args_parser.add_argument("--low", type=int, default=50, help="Canny low threshold (default: 50)")
    args_parser.add_argument("--high", type=int, default=150, help="Canny high threshold (default: 150)")
    args = args_parser.parse_args()
    if not 0 <= args.low < args.high <= 255:
        args_parser.error("thresholds must satisfy 0 <= --low < --high <= 255")
    if args.output and not args.image:
        args_parser.error("--output requires --image")
    if args.image:
        try:
            import cv2
        except ImportError:
            args_parser.exit(1, "Install dependencies: python -m pip install -r requirements-viewer.txt\n")
        image = cv2.imread(str(args.image))
        if image is None:
            args_parser.error(f"Cannot read image: {args.image}")
        output = args.output or args.image.with_name(args.image.stem + "_outline.png")
        try:
            save_outline(output, extract_outline(image, cv2, args.low, args.high), cv2)
        except (OSError, cv2.error) as error:
            args_parser.exit(1, f"{error}\n")
        print(f"Saved outline: {output}")
        return
    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        args_parser.exit(1, "Install dependencies: python -m pip install -r requirements-viewer.txt\n")
    if args.list_ports:
        for port in list_ports.comports():
            print(f"{port.device}: {port.description}")
        return
    if not args.port:
        args_parser.error("--port is required; use --list-ports to find the native USB port")
    try:
        import cv2
        import numpy as np
    except ImportError:
        args_parser.exit(1, "Install dependencies: python -m pip install -r requirements-viewer.txt\n")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    stop = threading.Event()
    lock = threading.Lock()
    latest = {}
    worker = threading.Thread(target=receive_frames,
                              args=(args.port, args.baudrate, stop, latest, lock, serial, cv2, np),
                              daemon=True)
    title = "ESP32-S3 OUTLINE | LEFT / RIGHT | S: save | Q or ESC: quit"
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        worker.start()
        while True:
            outlines = {}
            with lock:
                snapshot = latest.copy()
            canvas = np.zeros((520, 1280, 3), dtype=np.uint8)
            now = time.monotonic()
            for camera_id, name in enumerate(("LEFT", "RIGHT")):
                x = camera_id * 640
                item = snapshot.get(camera_id)
                label = f"{name}: waiting for frames"
                if item is not None:
                    image, frame_id, received_at = item
                    # Mount correction from original: LEFT 90 degrees CW, RIGHT 90 degrees CCW.
                    rotation = (cv2.ROTATE_90_CLOCKWISE if camera_id == 0
                                else cv2.ROTATE_90_COUNTERCLOCKWISE)
                    image = cv2.rotate(image, rotation)

                    outline = extract_outline(image, cv2, args.low, args.high)
                    outlines[camera_id] = outline
                    image = cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR)

                    height, width = image.shape[:2]
                    scale = min(640 / width, 480 / height)
                    resized = cv2.resize(image, (max(1, round(width * scale)),
                                                max(1, round(height * scale))))
                    h, w = resized.shape[:2]
                    left, top = x + (640 - w) // 2, 40 + (480 - h) // 2
                    canvas[top:top + h, left:left + w] = resized
                    age = now - received_at
                    label = f"{name}: frame {frame_id}  {width}x{height}"
                    if age > 2:
                        label += f"  STALE {age:.1f}s"
                cv2.putText(canvas, label, (x + 12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.imshow(title, canvas)
            key = cv2.waitKey(15) & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("s"), ord("S")):
                for camera_id, outline in outlines.items():
                    output = Path(("L_outline.png", "R_outline.png")[camera_id])
                    save_outline(output, outline, cv2)
                    logging.info("Saved outline: %s", output)
    except KeyboardInterrupt:
        pass
    except (cv2.error, OSError) as error:
        logging.error("OpenCV display failed (a desktop GUI and opencv-python are required): %s", error)
    finally:
        stop.set()
        if worker.is_alive():
            worker.join(timeout=2)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
