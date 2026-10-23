import threading
import re

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from .config import MAX_RECORD_SECONDS, SAMPLE_RATE, STT_MODEL
from .engine import speech_friendly_text


class Voice:
    def __init__(self):
        # CPU is the reliable default. Whisper CUDA needs compatible CUDA/cuDNN runtime DLLs.
        self.whisper = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
        self._tts_lock = threading.Lock()

    def record_until_release(self, stop_event):
        chunks = []
        error = []

        def callback(indata, frames, time_info, status):
            if status:
                # Overflow is recoverable; the audio can still be transcribed.
                pass
            if not stop_event.is_set():
                chunks.append(indata.copy())

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
                stop_event.wait(MAX_RECORD_SECONDS)
        except Exception as exc:
            error.append(exc)
        if error:
            raise RuntimeError(f"Не удалось открыть микрофон: {error[0]}") from error[0]
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0).reshape(-1)

    def transcribe(self, audio):
        if audio is None or len(audio) < SAMPLE_RATE // 3:
            return ""
        segments, _ = self.whisper.transcribe(audio, language="ru", vad_filter=True, beam_size=3)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def speak(self, text):
        # pyttsx3/Windows SAPI is COM-backed; create and use the engine in the same worker thread.
        try:
            import pyttsx3
            with self._tts_lock:
                engine = pyttsx3.init()
                try:
                    for voice in engine.getProperty("voices"):
                        desc = f"{voice.name} {voice.id}".casefold()
                        if "russian" in desc or "рус" in desc or "ru-ru" in desc:
                            engine.setProperty("voice", voice.id)
                            break
                    engine.setProperty("rate", 180)
                    engine.say(speech_friendly_text(text))
                    engine.runAndWait()
                finally:
                    engine.stop()
        except Exception:
            # The textual answer remains available even if Windows speech is unavailable.
            return
