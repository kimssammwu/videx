"""Shared-camera, persistent-mode VIDEX runtime, independent of its desktop UI."""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from pathlib import Path
import queue
import threading
import time
import uuid

import cv2
import numpy as np

from app.object_recognition.models import RecognitionConfig, RecognitionState
from app.object_recognition.recognizer import ObjectRecognizer
from .flood_fill import Detector
from .viewer import RESOLUTION_MODES, receive_frames

MODE_NAMES = {0: "보행 감지", 1: "글씨 읽기", 2: "물체 인식"}
# Both recognition modules request LEFT 640x480; application modes are distinct
# from firmware resolution commands.
MODE_RESOLUTIONS = {0: 0, 1: 1, 2: 1}
TERMINAL = (RecognitionState.IDLE, RecognitionState.COMPLETED, RecognitionState.ERROR)


class Worker:
    """One in-flight job, no frame backlog; closing never waits on HTTP."""
    def __init__(self, name):
        self.jobs = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.busy = False
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name=name)
        self.thread.start()

    def submit(self, generation, function):
        if self.busy or self.stop.is_set():
            return False
        self.busy = True
        self.jobs.put_nowait((generation, function))
        return True

    def _run(self):
        while not self.stop.is_set():
            try:
                generation, function = self.jobs.get(timeout=.1)
            except queue.Empty:
                continue
            if self.stop.is_set():
                return
            try:
                result, error = function(), None
            except Exception as exc:
                result, error = None, str(exc)
            self.results.put((generation, result, error))

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        return result

    def close(self):
        self.stop.set()
        self.thread.join(timeout=.2)


class CameraSource:
    """The only camera owner. All models consume snapshots of this source."""
    def __init__(self, *, port=None, camera=0, demo=False, baudrate=115200):
        self.port, self.camera, self.demo = port, camera, demo
        self.baudrate = baudrate
        self.latest = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.control = {"resolution": 0, "connected": False, "epoch": 0,
                        "status": "카메라 연결 중", "button_events": deque(maxlen=32)}
        self.thread = None

    def start(self):
        if self.port:
            import serial
            target = lambda: receive_frames(self.port, self.baudrate, self.stop, self.latest,
                                            self.lock, serial, cv2, np, self.control)
        else:
            target = self._local
        self.thread = threading.Thread(target=target, daemon=True, name="videx-camera")
        self.thread.start()

    def set_mode(self, mode):
        with self.lock:
            self.control["resolution"] = MODE_RESOLUTIONS[mode]
            self.latest.clear()

    def snapshot(self):
        with self.lock:
            events = list(self.control["button_events"])
            self.control["button_events"].clear()
            return self.latest.copy(), dict(self.control), events

    def _local(self):
        capture = None
        index = 0
        started = time.monotonic()
        try:
            while not self.stop.is_set():
                if self.demo:
                    image = demo_image(index, int((time.monotonic() - started) / 3) % 2)
                else:
                    if capture is None:
                        capture = cv2.VideoCapture(self.camera)
                    ok, image = capture.read()
                    if not ok:
                        capture.release()
                        capture = None
                        with self.lock:
                            self.latest.clear()
                            self.control.update(connected=False, status="웹캠 연결 실패; 1초 후 재시도")
                        self.stop.wait(1)
                        continue
                now = time.monotonic()
                with self.lock:
                    if not self.control["connected"]:
                        self.control["epoch"] += 1
                    self.control.update(connected=True, status="DEMO / 모의 인식" if self.demo else "노트북 웹캠")
                    self.latest[0] = (image, index, now, None, False, None, False)
                index += 1
                self.stop.wait(.1 if self.demo else .025)
        finally:
            if capture is not None:
                capture.release()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)


def demo_image(index, side=0):
    image = np.full((480, 640, 3), 200 if side == 0 else 65, np.uint8)
    color = (30, 30, 30) if side == 0 else (230, 230, 230)
    cv2.rectangle(image, (175, 65), (465, 405), color, 5)
    for y in range(100, 390, 30):
        cv2.line(image, (195, y), (445, y), color, 3)
    cv2.putText(image, "VIDEX FRONT" if side == 0 else "VIDEX BACK", (190, 250),
                cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
    return image


class WalkingPipeline:
    def __init__(self):
        self.flood = Detector()
        self.last_id = None

    def process(self, image, frame_id, received_at):
        if self.last_id is not None and frame_id <= self.last_id:
            self.flood = Detector()
        self.last_id = frame_id
        flood = self.flood.process(image, received_at, frame_id=frame_id)
        overlay = image.copy()
        if flood.valid and flood.analysis is not None:
            mask = flood.analysis["unfilled"]
            overlay[mask] = (overlay[mask] * .5 + np.array([0, 0, 255]) * .5).astype(np.uint8)
        return {"kind": "walk", "state": flood.state, "reason": flood.reason,
                "unfilled_ratio": flood.unfilled_ratio, "central_ratio": flood.central_ratio,
                "clears": flood.consecutive_clears,
                "confirm": self.flood.config.confirm,
                "received_at": received_at, "image": overlay}


class Runtime:
    """UI-thread controller. Mode lifetime is separate from recognition lifetime."""
    def __init__(self, source, text_client, object_client, *, output_dir, speech=None,
                 mode=0, rotation=None, object_config=None):
        self.source, self.text_client, self.object_client = source, text_client, object_client
        self.output_dir = Path(output_dir)
        self.speech = speech
        self.rotation = ("cw90" if source.port else "none") if rotation is None else rotation
        self.object_config = object_config or RecognitionConfig()
        self.vision = Worker("videx-recognition")
        self.walker = Worker("videx-walking")
        self.messages = queue.Queue(maxsize=64)
        self.generation = 0
        self.source_epoch = None
        self.closed = False
        self.frame = None
        self.snapshot = {}
        self.transport = {}
        self.history = deque(maxlen=100)
        self.select_mode(mode)

    def say(self, text):
        print(text, flush=True)
        if self.speech:
            self.speech.say(text)

    def select_mode(self, mode):
        if mode not in MODE_NAMES:
            raise ValueError("mode must be 0, 1 or 2")
        self.generation += 1
        self.mode = mode
        self.changed_at = time.monotonic()
        self.source.set_mode(mode)
        self.snapshot = {}
        self.frame = None
        self.last_token = None
        self.last_walk_at = None
        self.overlay = None
        self.result = None
        self.error = None
        self.state = "READY"
        self.guidance = "촬영 버튼 또는 Space를 누르세요."
        self.walk_result = None
        self.warning_latched = False
        self.pipeline = WalkingPipeline()
        self.recognizer = None
        if self.speech:
            self.speech.reset()
        self.say(MODE_NAMES[mode] + " 모드입니다.")
        if mode == 0:
            self.guidance = "새 영상을 받으면 보행 감지를 계속 실행합니다."
        self.history.append({"event": "mode", "mode": mode})

    def _announce(self, generation, text):
        try:
            self.messages.put_nowait((generation, text))
        except queue.Full:
            pass

    def _upright(self, image):
        rotations = {"cw90": cv2.ROTATE_90_CLOCKWISE, "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
                     "180": cv2.ROTATE_180}
        return cv2.rotate(image, rotations[self.rotation]) if self.rotation in rotations else image

    def _current_frame(self, now):
        item = self.snapshot.get(0)
        if item is None or item[6] or item[0] is None:
            return None
        if now - item[2] > 2 or item[2] < self.changed_at:
            return None
        if self.source.port and item[0].shape[1::-1] != RESOLUTION_MODES[MODE_RESOLUTIONS[self.mode]]:
            return None
        return item

    def action(self):
        if self.mode == 0:
            self.say(self.guidance)
            return False
        if self.vision.busy:
            self.guidance = "이전 인식 요청이 처리 중입니다. 잠시 기다려 주세요."
            return False
        item = self._current_frame(time.monotonic())
        if item is None:
            self.guidance = "현재 모드의 새 카메라 영상을 기다리고 있습니다."
            self.say(self.guidance)
            return False
        self.error = None
        image = self._upright(item[0]).copy()
        generation = self.generation
        if self.mode == 1:
            self.state = "RECOGNIZING"
            self.guidance = "글씨를 인식하고 있습니다."
            text_client = self.text_client
            def recognize():
                self.output_dir.mkdir(parents=True, exist_ok=True)
                path = self.output_dir / ("text_" + uuid.uuid4().hex + ".jpg")
                ok, encoded = cv2.imencode(".jpg", image)
                if not ok:
                    raise ValueError("JPEG 저장 실패")
                jpeg = encoded.tobytes()
                path.write_bytes(jpeg)
                result = text_client.recognize(jpeg)
                if not isinstance(result, dict) or set(result) != {"text"} or not isinstance(result["text"], str):
                    raise ValueError("글씨 인식 응답 형식 오류")
                return {"kind": "text", "result": result["text"], "path": str(path)}
            return self.vision.submit(generation, recognize)
        if self.recognizer is None or self.recognizer.state in TERMINAL or self.state == "ERROR":
            config = replace(self.object_config, output_dir=self.output_dir / ("object_" + uuid.uuid4().hex))
            self.recognizer = ObjectRecognizer(config, self.object_client,
                                              lambda text: self._announce(generation, text))
            self.recognizer.start_recognition("LEFT")
        return self._submit_object(item, manual=True)

    def _submit_object(self, item, manual=False):
        recognizer = self.recognizer
        image = self._upright(item[0]).copy()
        def process():
            metrics = recognizer.process_frame(image, camera_id="LEFT", frame_id=item[1],
                                               timestamp=item[2], manual_capture=manual)
            return {"kind": "object", **recognizer.get_recognition_state(), "reason": metrics.reason}
        if self.vision.submit(self.generation, process):
            self.last_token = (item[1], item[2])
            self.state = recognizer.state.value
            return True
        return False

    def tick(self, now=None):
        if self.closed:
            return
        now = time.monotonic() if now is None else now
        self.snapshot, self.transport, events = self.source.snapshot()
        epoch = self.transport.get("epoch", 0)
        if self.source_epoch not in (None, 0) and epoch != self.source_epoch:
            self.select_mode(self.mode)
        self.source_epoch = epoch
        for event, received_at in events:
            if now - received_at > 1:
                continue
            if event == "LONG":
                self.select_mode((self.mode + 1) % 3)
            elif event == "SHORT":
                self.action()
        for worker in (self.walker, self.vision):
            completed = worker.poll()
            if completed is not None:
                generation, value, error = completed
                if generation == self.generation:
                    self._accept(value, error, now)
        while True:
            try:
                generation, text = self.messages.get_nowait()
            except queue.Empty:
                break
            if generation == self.generation:
                self.guidance = text
                self.say(text)
        item = self._current_frame(now)
        self.frame = self._upright(item[0]) if item is not None else None
        if self.mode == 0:
            if item is None or self.last_walk_at is None or now - self.last_walk_at > 2:
                self.state = "UNKNOWN"
                self.overlay = None
                self.guidance = "보행 판단 대기: 새 영상 또는 처리 결과가 없습니다."
            if item is not None and not self.walker.busy and self.last_token != (item[1], item[2]):
                pipeline = self.pipeline
                frame = self.frame.copy()
                self.walker.submit(self.generation, lambda: pipeline.process(frame, item[1], item[2]))
                self.last_token = (item[1], item[2])
        elif self.mode == 2 and self.recognizer is not None and not self.vision.busy:
            if self.recognizer.state not in TERMINAL and self.state != "ERROR":
                if self.recognizer.tick(now):
                    self.state = "ERROR"
                    self.error = self.recognizer.error
                elif item is not None and self.last_token != (item[1], item[2]):
                    self._submit_object(item)

    def _accept(self, value, error, now):
        if error is not None:
            self.state, self.error = "ERROR", error
            self.guidance = "처리 실패. 촬영 버튼으로 다시 시도할 수 있습니다."
            self.say(self.guidance)
            return
        if value["kind"] == "walk":
            if now - value["received_at"] > 2:
                return
            self.walk_result = value
            self.last_walk_at = value["received_at"]
            self.state = value["state"]
            self.overlay = value["image"]
            self.guidance = {"WARNING": "전방에 장애물 징후가 있습니다.",
                             "UNKNOWN": "보행 판단에 필요한 정보를 수집 중입니다.",
                             "MONITORING": "보행 감지 중입니다."}[self.state]
            if self.state == "WARNING" and not self.warning_latched:
                self.say(self.guidance)
                self.warning_latched = True
            elif self.state == "MONITORING" and value["clears"] >= value["confirm"]:
                self.warning_latched = False
        elif value["kind"] == "text":
            self.state, self.result = "COMPLETED", value["result"]
            self.guidance = "글씨 인식 완료. 다시 촬영할 수 있습니다."
            self.history.append(value)
            self.say(self.result or "인식된 글자가 없습니다.")
        else:
            self.state, self.result, self.error = value["state"], value["result"], value["error"]
            if value["reason"] == "duplicate_rejected":
                self.guidance = "앞면과 같습니다. 물체를 뒤집어서 뒷면을 보여주세요."
            if self.state in ("COMPLETED", "ERROR"):
                self.history.append(value)

    def status(self):
        return {"mode": self.mode, "mode_name": MODE_NAMES[self.mode], "state": self.state,
                "guidance": self.guidance, "result": self.result, "error": self.error,
                "camera": self.transport.get("status"), "mock": getattr(self.text_client, "calls", None) is not None,
                "vision_busy": self.vision.busy, "history": list(self.history),
                "walking": {k: v for k, v in (self.walk_result or {}).items() if k != "image"}}

    def close(self):
        self.closed = True
        self.generation += 1
        self.source.close()
        self.walker.close()
        self.vision.close()
        if self.speech:
            self.speech.close()
