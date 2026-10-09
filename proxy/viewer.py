#!/usr/bin/env python3
"""Display CAM2 JPEG frames from the ESP32-S3 native USB CDC port."""

import argparse
from dataclasses import dataclass
import logging
import struct
import threading
import time
from typing import Optional


HEADER = struct.Struct("<IBBBBIIHHQ")
MAGIC = b"CAM2"
BUTTON_PRESSED = 1 << 0
BUTTON_VALID = 1 << 1
BUTTON_CLASSIFIED = 1 << 2
STREAM_PAUSED = 1 << 3
BUTTON_RESULTS = {0: None, 1: "SHORT", 2: "LONG"}
MAX_JPEG_SIZE = 1024 * 1024
MAX_FRAME_DIMENSION = 4096
FRAME_TIMEOUT = 6.0
RESOLUTION_MODES = {0: (320, 240), 1: (640, 480), 2: (1280, 1024)}


def resolution_label(mode):
    size = RESOLUTION_MODES[mode]
    return f"{size[0]}x{size[1]} ({'both cameras' if mode == 0 else 'LEFT only'})"


def resolution_command(mode):
    if mode not in RESOLUTION_MODES:
        raise ValueError("Resolution mode must be 0, 1, or 2")
    return bytes([mode])


@dataclass(frozen=True)
class Frame:
    camera_id: int
    frame_id: int
    width: int
    height: int
    timestamp_us: int
    button_pressed: Optional[bool]
    button_classified: bool
    button_result: Optional[str]
    jpeg: bytes
    paused: bool = False


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
            _, camera_id, flags, result, reserved, frame_id, size, width, height, timestamp = HEADER.unpack_from(self.buffer)
            paused = (camera_id == 1 and flags == STREAM_PAUSED and result == 0
                      and reserved == 0 and size == 0 and width == 0 and height == 0)
            if not (paused or (
                camera_id in (0, 1)
                and reserved == 0
                and (
                    (flags == 0 and result == 0)
                    or (camera_id == 0 and flags in (2, 3) and result == 0)
                    or (camera_id == 0 and flags == BUTTON_CLASSIFIED and result == 0)
                    or (camera_id == 0 and flags in (6, 7) and result in BUTTON_RESULTS)
                )
                and 4 <= size <= MAX_JPEG_SIZE
                and 0 < width <= MAX_FRAME_DIMENSION
                and 0 < height <= MAX_FRAME_DIMENSION
            )):
                del self.buffer[0]
                self.started = None
                continue
            end = HEADER.size + size
            if len(self.buffer) < end:
                break
            jpeg = bytes(self.buffer[HEADER.size:end])
            if not paused and not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
                del self.buffer[0]
                self.started = None
                continue
            button_pressed = bool(flags & BUTTON_PRESSED) if flags & BUTTON_VALID else None
            frames.append(Frame(camera_id, frame_id, width, height, timestamp, button_pressed,
                                bool(flags & BUTTON_CLASSIFIED), BUTTON_RESULTS[result], jpeg, paused))
            del self.buffer[:end]
            self.started = None
        return frames


def receive_frames(port, baudrate, stop, latest, lock, serial, cv2, np, control):
    """Keep serial reads and JPEG decoding off the GUI thread."""
    while not stop.is_set():
        try:
            with serial.Serial(port, baudrate=baudrate, timeout=0.1, write_timeout=0.5) as connection:
                logging.info("USB connected: %s", port)
                parser = FrameParser()
                sent_mode = None
                while not stop.is_set():
                    with lock:
                        mode = control["resolution"]
                    if mode != sent_mode:
                        if connection.write(resolution_command(mode)) != 1:
                            raise serial.SerialException("Incomplete resolution command write")
                        sent_mode = mode
                        logging.info("Requested resolution mode %d (%s)", mode,
                                     resolution_label(mode))
                    data = connection.read(min(max(connection.in_waiting, 1), 65536))
                    # Empty reads also let the parser expire incomplete frames.
                    for frame in parser.feed(data):
                        received_at = time.monotonic()
                        image = None
                        if not frame.paused:
                            image = cv2.imdecode(np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if image is None or image.shape[:2] != (frame.height, frame.width):
                                logging.warning("Invalid JPEG or dimensions: camera %d frame %d",
                                                frame.camera_id, frame.frame_id)
                                continue
                        with lock:
                            latest[frame.camera_id] = (image, frame.frame_id, received_at,
                                                      frame.button_pressed, frame.button_classified,
                                                      frame.button_result, frame.paused)
        except (serial.SerialException, OSError) as error:
            logging.warning("USB unavailable: %s; retrying in 1 second", error)
            stop.wait(1.0)


def main():
    args_parser = argparse.ArgumentParser(description=__doc__)
    args_parser.add_argument("--port", help="Native USB port, e.g. /dev/ttyACM0 or COM5")
    args_parser.add_argument("--list-ports", action="store_true", help="List ports and exit")
    args_parser.add_argument("--baudrate", type=int, default=115200,
                             help="CDC line setting (native USB speed is independent of baudrate)")
    args_parser.add_argument("--resolution", type=int, choices=RESOLUTION_MODES, default=0,
                             help="0=both cameras 320x240; 1=LEFT 640x480; 2=LEFT 1280x1024 (RIGHT paused)")
    args = args_parser.parse_args()
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
    control = {"resolution": args.resolution}
    worker = threading.Thread(target=receive_frames,
                              args=(args.port, args.baudrate, stop, latest, lock, serial, cv2, np, control),
                              daemon=True)
    title = "ESP32-S3 | 0:both 320 1:LEFT 640 2:LEFT 1280 | Q or ESC to quit"
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        worker.start()
        while True:
            with lock:
                snapshot = latest.copy()
                requested_mode = control["resolution"]
            canvas = np.zeros((550, 1280, 3), dtype=np.uint8)
            now = time.monotonic()
            for camera_id, name in enumerate(("LEFT", "RIGHT")):
                x = camera_id * 640
                item = snapshot.get(camera_id)
                label = f"{name}: waiting for frames"
                if item is not None:
                    image, frame_id, received_at, button_pressed, button_classified, button_result, paused = item
                    if paused:
                        label = f"{name}: PAUSED (LEFT-only mode)"
                    else:
                        height, width = image.shape[:2]
                        scale = min(640 / width, 480 / height)
                        resized = cv2.resize(image, (max(1, round(width * scale)),
                                                    max(1, round(height * scale))))
                        h, w = resized.shape[:2]
                        left, top = x + (640 - w) // 2, 40 + (480 - h) // 2
                        canvas[top:top + h, left:left + w] = resized
                        if camera_id == 0:
                            button_text = ("BUTTON: UNKNOWN" if button_pressed is None else
                                           "BUTTON: PRESSED" if button_pressed else "BUTTON: RELEASED")
                            if button_classified:
                                button_text += f"  S3: {button_result or '--'}"
                            else:
                                button_text += "  RAW (update S3)"
                            cv2.rectangle(canvas, (x, 40), (x + 640, 74), (0, 0, 0), -1)
                            cv2.putText(canvas, button_text, (x + 12, 64),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                        (0, 255, 0) if button_pressed else (200, 200, 200),
                                        1, cv2.LINE_AA)
                        label = f"{name}: frame {frame_id}  {width}x{height}"
                    age = now - received_at
                    if age > 2:
                        label += f"  STALE {age:.1f}s"
                cv2.putText(canvas, label, (x + 12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Requested: {resolution_label(requested_mode)} | keys 0 / 1 / 2 | actual sizes above",
                        (12, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow(title, canvas)
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("0"), ord("1"), ord("2")):
                with lock:
                    control["resolution"] = key - ord("0")
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    except cv2.error as error:
        logging.error("OpenCV display failed (a desktop GUI and opencv-python are required): %s", error)
    finally:
        stop.set()
        if worker.is_alive():
            worker.join(timeout=2)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
