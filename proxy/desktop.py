"""Native Python desktop entry point. Camera, models, and speech share one process."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import time

import cv2

from app.object_recognition.api import HttpVisionClient as ObjectAPI, MockVisionClient as MockObject
from app.text_recognition.vision import HttpVisionClient as TextAPI, MockVisionClient as MockText
from .local_speech import LocalSpeech
from .runtime import CameraSource, MODE_NAMES, Runtime

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--port", help="ESP32 native USB, e.g. COM5")
    source.add_argument("--camera", type=int, help="Laptop webcam index (default: 0)")
    source.add_argument("--demo", action="store_true", help="Synthetic camera AND mock recognition")
    p.add_argument("--list-ports", action="store_true")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--mode", type=int, choices=(0, 1, 2), help="0=walk, 1=text, 2=object; default: last mode")
    p.add_argument("--rotation", choices=("none", "cw90", "ccw90", "180"),
                   help="Default: USB LEFT cw90, webcam none")
    p.add_argument("--mock", action="store_true", help="Use explicit mock results with a real camera")
    p.add_argument("--model", default=os.environ.get("VIDEX_VISION_MODEL", "gpt-4.1-mini"))
    p.add_argument("--mute", action="store_true", help="Disable local laptop speech")
    p.add_argument("--headless", action="store_true", help="Run without the desktop window")
    p.add_argument("--seconds", type=float, default=0, help="0=until quit")
    p.add_argument("--output-dir", type=Path, default=ROOT / "desktop-output")
    return p


def saved_mode(output_dir):
    try:
        mode = json.loads((output_dir / "settings.json").read_text(encoding="utf-8"))["mode"]
        return mode if type(mode) is int and mode in MODE_NAMES else 0
    except (OSError, ValueError, KeyError, TypeError):
        return 0


class Desktop:
    def __init__(self, runtime, args):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.runtime, self.args = tk, runtime, args
        self.root = root = tk.Tk()
        root.title("VIDEX · 노트북 독립 실행")
        root.geometry("1120x820")
        root.minsize(900, 680)
        self.closed = False
        self.started = time.monotonic()
        self.photo = None
        self.last_text = None
        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="VIDEX  |  카메라 인식 · 음성 안내", font=("맑은 고딕", 19, "bold")).pack(anchor="w")
        self.api_status = tk.StringVar()
        ttk.Label(outer, textvariable=self.api_status).pack(anchor="w", pady=(4, 10))
        connection = ttk.Frame(outer)
        connection.pack(fill="x", pady=(0, 10))
        ttk.Label(connection, text="카메라").pack(side="left")
        choices = ["웹캠 0", "웹캠 1", "DEMO"]
        try:
            from serial.tools import list_ports
            choices.extend(port.device for port in list_ports.comports())
        except ImportError:
            pass
        selected = args.port or ("DEMO" if args.demo else f"웹캠 {args.camera or 0}")
        self.camera = tk.StringVar(value=selected)
        ttk.Combobox(connection, textvariable=self.camera, values=choices, width=25).pack(side="left", padx=8)
        ttk.Button(connection, text="연결 / 재연결", command=self.connect).pack(side="left")
        self.camera_status = tk.StringVar()
        ttk.Label(connection, textvariable=self.camera_status).pack(side="left", padx=12)
        modes = ttk.Frame(outer)
        modes.pack(fill="x", pady=(0, 10))
        self.mode = tk.IntVar(value=runtime.mode)
        for value, name in MODE_NAMES.items():
            ttk.Radiobutton(modes, text=f"{value}  {name}", variable=self.mode, value=value,
                            command=lambda v=value: runtime.select_mode(v)).pack(side="left", padx=(0, 24))
        content = ttk.Frame(outer)
        content.pack(fill="both", expand=True)
        preview = ttk.Frame(content)
        preview.pack(side="left", fill="both", expand=True)
        self.image_label = tk.Label(preview, bg="#18212b", fg="white", text="카메라 영상을 기다리고 있습니다.")
        self.image_label.pack(fill="both", expand=True)
        self.diagnostics = tk.StringVar()
        ttk.Label(preview, textvariable=self.diagnostics, wraplength=680).pack(anchor="w", pady=8)
        panel = ttk.Frame(content, padding=(16, 0, 0, 0), width=320)
        panel.pack(side="right", fill="y")
        self.state = tk.StringVar()
        ttk.Label(panel, textvariable=self.state, font=("맑은 고딕", 15, "bold")).pack(anchor="w")
        self.guidance = tk.StringVar()
        ttk.Label(panel, textvariable=self.guidance, wraplength=290).pack(anchor="w", pady=10)
        ttk.Button(panel, text="촬영 / 앞·뒷면 확정 (Space)", command=runtime.action).pack(fill="x", pady=4)
        ttk.Button(panel, text="현재 모드 다시 시작 (R)", command=lambda: runtime.select_mode(runtime.mode)).pack(fill="x", pady=4)
        ttk.Button(panel, text="결과 다시 읽기 (V)", command=self.repeat).pack(fill="x", pady=4)
        ttk.Label(panel, text="인식 결과 / 오류").pack(anchor="w", pady=(16, 4))
        self.result = tk.Text(panel, width=31, height=17, wrap="word", font=("맑은 고딕", 11), state="disabled")
        self.result.pack(fill="both", expand=True)
        self.speech_status = tk.StringVar()
        ttk.Label(panel, textvariable=self.speech_status, wraplength=290).pack(anchor="w", pady=8)
        ttk.Label(outer, text="0/1/2 모드 전환 · Space 촬영 · R 다시 시작 · V 다시 읽기 · Q/Esc 종료\n"
                  "ESP32 버튼: 길게 누른 뒤 떼면 다음 모드, 짧게 누른 뒤 떼면 촬영/확정 · 보행 감지는 실험 단계입니다.").pack(anchor="w", pady=(10, 0))
        root.bind("<KeyPress>", self.key)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def connect(self):
        value = self.camera.get().strip()
        if not value:
            return
        demo = value == "DEMO"
        camera = 0
        if value.startswith("웹캠 "):
            try:
                camera = int(value.split()[1])
            except (ValueError, IndexError):
                self.runtime.guidance = "웹캠 번호를 확인해 주세요. 예: 웹캠 0"
                return
        port = None if demo or value.startswith("웹캠 ") else value
        # A real camera never inherits mock clients just because DEMO was selected earlier.
        mock = demo or self.args.mock
        self.runtime.text_client, self.runtime.object_client = clients(self.args, mock)
        self.runtime.source.close()
        self.runtime.source = CameraSource(port=port, camera=camera, demo=demo, baudrate=self.args.baudrate)
        self.runtime.rotation = self.args.rotation or ("cw90" if port else "none")
        self.runtime.source_epoch = None
        self.runtime.select_mode(self.runtime.mode)
        self.runtime.source.start()

    def repeat(self):
        self.runtime.say(self.runtime.result or self.runtime.guidance)

    def key(self, event):
        if event.widget.winfo_class() in ("TCombobox", "Text", "Entry", "TEntry"):
            return
        key = event.keysym.lower()
        if key in ("0", "1", "2"):
            self.runtime.select_mode(int(key))
        elif key == "space":
            self.runtime.action()
        elif key == "r":
            self.runtime.select_mode(self.runtime.mode)
        elif key == "v":
            self.repeat()
        elif key in ("q", "escape"):
            self.close()
        return "break"

    def refresh(self):
        if self.closed:
            return
        r = self.runtime
        r.tick()
        self.mode.set(r.mode)
        self.state.set(f"{MODE_NAMES[r.mode]} · {r.state}")
        self.guidance.set(r.guidance)
        self.camera_status.set(r.transport.get("status", "연결 중"))
        is_mock = hasattr(r.text_client, "calls")
        self.api_status.set("모의 인식 사용 중 — 결과는 테스트 문구입니다." if is_mock else
                            "실제 Vision API 사용 · 글씨/물체 인식에는 인터넷과 VIDEX_VISION_API_KEY가 필요합니다.")
        text = ("[모의 인식]\n" if is_mock else "") + (r.error or r.result or "아직 인식 결과가 없습니다.")
        if text != self.last_text:
            self.result.configure(state="normal")
            self.result.delete("1.0", "end")
            self.result.insert("1.0", text)
            self.result.configure(state="disabled")
            self.last_text = text
        self.speech_status.set(r.speech.error or ("음성 꺼짐" if not r.speech.enabled else "노트북 음성 출력 사용"))
        walking = r.walk_result or {}
        detail = ""
        if r.mode == 0:
            detail = f"Flood Fill: {r.state}"
            if walking.get("unfilled_ratio") is not None:
                detail += f"  |  미충전 {walking['unfilled_ratio']:.1%}  |  중앙 {walking['central_ratio']:.1%}"
            detail += "\n" + walking.get("reason", "새 영상을 기다리고 있습니다.")
        self.diagnostics.set(detail)
        frame = r.overlay if r.mode == 0 and r.overlay is not None else r.frame
        if frame is not None:
            h, w = frame.shape[:2]
            scale = min(700 / w, 500 / h)
            resized = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
            # Tk supports binary PPM directly, including with opencv-python-headless.
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            data = f"P6\n{w} {h}\n255\n".encode("ascii") + rgb.tobytes()
            self.photo = self.tk.PhotoImage(data=data, format="PPM")
            self.image_label.configure(image=self.photo, text="")
        else:
            self.image_label.configure(image="", text="새 영상을 기다리고 있습니다.\n카메라 연결과 선택한 포트를 확인해 주세요.")
            self.photo = None
        if self.args.seconds and time.monotonic() - self.started >= self.args.seconds:
            self.close()
            return
        self.root.after(40, self.refresh)

    def run(self):
        self.refresh()
        self.root.mainloop()

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()


def clients(args, mock):
    if mock:
        return MockText("VIDEX 데모 글씨입니다."), MockObject("VIDEX 데모 제품")
    endpoint = os.environ.get("VIDEX_VISION_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    return TextAPI(model=args.model, endpoint=endpoint, timeout=20), ObjectAPI(model=args.model, endpoint=endpoint, timeout=20)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds < 0:
        p.error("--seconds must be finite and nonnegative")
    if args.list_ports:
        from serial.tools import list_ports
        for port in list_ports.comports():
            print(f"{port.device}: {port.description}")
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    try:
        text_client, object_client = clients(args, args.mock or args.demo)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    mode = saved_mode(args.output_dir) if args.mode is None else args.mode
    source = CameraSource(port=args.port, camera=args.camera or 0, demo=args.demo, baudrate=args.baudrate)
    speech = LocalSpeech(enabled=not args.mute)
    runtime = Runtime(source, text_client, object_client, output_dir=args.output_dir,
                      speech=speech, mode=mode, rotation=args.rotation)
    code = 0
    try:
        source.start()
        if args.headless:
            started = time.monotonic()
            previous = None
            while not args.seconds or time.monotonic() - started < args.seconds:
                runtime.tick()
                current = (runtime.mode, runtime.state, runtime.error)
                if current != previous:
                    logging.info("mode=%s state=%s error=%s", *current)
                    previous = current
                time.sleep(.02)
        else:
            Desktop(runtime, args).run()
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("VIDEX 실행 실패")
        code = 1
    finally:
        runtime.close()
        status = runtime.status()
        try:
            (args.output_dir / "session.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            (args.output_dir / "settings.json").write_text(json.dumps({"mode": runtime.mode}), encoding="utf-8")
        except OSError as exc:
            logging.error("결과 저장 실패: %s", exc)
            code = 1
        print(json.dumps(status, ensure_ascii=False))
    return code
