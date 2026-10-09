"""Offline laptop speech. No browser, HTTP service, or phone is involved."""
from __future__ import annotations

import logging
import queue
import threading


class LocalSpeech:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.error = None
        self.generation = 0
        self._queue = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._thread = None
        if enabled:
            self._thread = threading.Thread(target=self._run, daemon=True, name="videx-local-speech")
            self._thread.start()

    def reset(self):
        self.generation += 1

    def say(self, text):
        if not self.enabled or self._stop.is_set():
            return
        item = (self.generation, str(text))
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass

    def _run(self):
        engine = None
        try:
            # Initialize and operate SAPI on the SAME background thread.
            import pyttsx3
            engine = pyttsx3.init()
            for voice in engine.getProperty("voices"):
                description = str((voice.id, voice.name, voice.languages)).lower()
                if any(token in description for token in ("korean", "ko-kr", "ko_kr", "heami")):
                    engine.setProperty("voice", voice.id)
                    break
            engine.setProperty("rate", 175)
            while not self._stop.is_set():
                try:
                    generation, text = self._queue.get(timeout=.1)
                except queue.Empty:
                    continue
                if generation != self.generation:
                    continue
                engine.say(text)
                engine.startLoop(False)
                try:
                    while engine.isBusy() and not self._stop.is_set() and generation == self.generation:
                        engine.iterate()
                        self._stop.wait(.02)
                    engine.stop()
                finally:
                    engine.endLoop()
        except Exception as exc:
            self.error = "음성 출력 불가: " + str(exc)
            logging.warning("%s", self.error)
        finally:
            if engine is not None:
                engine.stop()

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
