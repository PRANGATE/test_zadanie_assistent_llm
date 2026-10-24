import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from .engine import AssistantEngine
from .llm import LocalLLM
from .voice import Voice


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Голосовой ассистент задач")
        self.root.geometry("820x680")
        self.root.minsize(680, 540)
        self.stop_event = threading.Event()
        self.recording = False
        self.busy = True
        self.voice = None
        self.engine = None
        self._state_lock = threading.Lock()
        self._ui_queue = queue.Queue()
        self._closed = False
        self.root.after(50, self._drain_ui_queue)

        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Локальный голосовой ассистент", font=("Segoe UI", 18, "bold")).pack(pady=(0, 6))
        ttk.Label(outer, text="Зажмите кнопку, произнесите запрос и отпустите её.").pack(pady=(0, 12))
        self.status = tk.StringVar(value="Загружаю локальные модели. Первый запуск может занять несколько минут…")
        ttk.Label(outer, textvariable=self.status, wraplength=760).pack(fill="x", pady=(0, 12))

        self.mic = tk.Button(outer, text="ЗАЖМИТЕ И ГОВОРИТЕ", font=("Segoe UI", 14, "bold"),
                             height=3, width=28, relief="raised", bg="#d9eaff")
        self.mic.pack(pady=8)
        self.mic.bind("<ButtonPress-1>", self.start_recording)
        self.mic.bind("<ButtonRelease-1>", self.stop_recording)
        self.mic.configure(state="disabled")

        ttk.Label(outer, text="Или введите запрос текстом:").pack(anchor="w", pady=(16, 4))
        row = ttk.Frame(outer); row.pack(fill="x")
        self.text_input = ttk.Entry(row)
        self.text_input.pack(side="left", fill="x", expand=True)
        self.text_input.bind("<Return>", lambda _event: self.submit_text())
        self.send_button = ttk.Button(row, text="Отправить", command=self.submit_text, state="disabled")
        self.send_button.pack(side="left", padx=(8, 0))

        ttk.Label(outer, text="История диалога", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(16, 4))
        self.history = tk.Text(outer, height=16, wrap="word", state="disabled", font=("Segoe UI", 10))
        self.history.pack(fill="both", expand=True)
        self._append("Ассистент", "Запускаю локальные модели. Данные хранятся на этом компьютере.")
        threading.Thread(target=self._init_models, daemon=True, name="model-loader").start()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _ui(self, fn, *args, **kwargs):
        """Queue UI updates; only the Tk main thread touches Tk widgets."""
        if not self._closed:
            self._ui_queue.put((fn, args, kwargs))

    def _drain_ui_queue(self):
        if self._closed:
            return
        while True:
            try:
                fn, args, kwargs = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn(*args, **kwargs)
            except tk.TclError:
                # The window may have been closed while a worker was finishing.
                continue
        try:
            self.root.after(50, self._drain_ui_queue)
        except tk.TclError:
            self._closed = True

    def _init_models(self):
        try:
            self._ui(self._set_status, "Загружаю языковую модель Qwen2.5-1.5B-Instruct…")
            llm = LocalLLM()
            self._ui(self._set_status, "Загружаю распознавание речи…")
            voice = Voice()
            engine = AssistantEngine(llm)
            self.llm, self.voice, self.engine = llm, voice, engine
            self._ui(self._set_ready)
            self._ui(self._append, "Ассистент", "Готов к работе.")
        except Exception as exc:
            self._ui(self._set_status, "Не удалось загрузить модели. Подробности в сообщении об ошибке.")
            self._ui(messagebox.showerror, "Ошибка загрузки моделей", f"{type(exc).__name__}: {exc}")
            self._ui(self._set_busy, False)

    def _set_status(self, text):
        self.status.set(text)

    def _set_ready(self):
        with self._state_lock:
            self.busy = False
        self.send_button.configure(state="normal")
        self.mic.configure(state="normal")
        self.status.set("Готово. Можно вводить следующий запрос.")

    def _set_busy(self, busy):
        with self._state_lock:
            self.busy = busy
        enabled = bool(self.engine) and not busy
        self.send_button.configure(state="normal" if enabled else "disabled")
        self.mic.configure(state="normal" if enabled else "disabled")

    def _append(self, who, text):
        self.history.configure(state="normal")
        self.history.insert("end", f"{who}: {text}\n\n")
        self.history.see("end")
        self.history.configure(state="disabled")

    def _acquire_busy(self):
        with self._state_lock:
            if self.busy:
                return False
            self.busy = True
            return True

    def start_recording(self, _event=None):
        if not self.voice or not self.engine or not self._acquire_busy():
            self.status.set("Модели загружаются или обрабатывается предыдущий запрос.")
            return "break"
        self.recording = True
        self.stop_event.clear()
        self.send_button.configure(state="disabled")
        self.mic.configure(text="ЗАПИСЬ… ОТПУСТИТЕ ДЛЯ ОТПРАВКИ", bg="#ffcccc")
        self.status.set("Идёт запись…")
        threading.Thread(target=self._record_worker, daemon=True, name="audio-capture").start()
        return "break"

    def _record_worker(self):
        try:
            audio = self.voice.record_until_release(self.stop_event)
            if audio is None:
                self._ui(self._set_status, "Звук не записан. Проверьте микрофон и попробуйте снова.")
                return
            self._ui(self._set_status, "Распознаю речь…")
            text = self.voice.transcribe(audio)
            if not text:
                self._ui(self._set_status, "Речь не распознана. Попробуйте ещё раз.")
                return
            self._ui(self._append, "Вы (голос)", text)
            self._run_request(text)
        except Exception as exc:
            self._ui(self._append, "Ошибка микрофона", f"{type(exc).__name__}: {exc}")
            self._ui(self._set_status, "Ошибка записи или распознавания. Можно попробовать текстовый ввод.")
        finally:
            self.recording = False
            self._ui(self._request_finished)

    def stop_recording(self, _event=None):
        if self.recording:
            self.stop_event.set()
            self.mic.configure(text="ОБРАБОТКА…", bg="#ffe6b3")
            self.status.set("Обрабатываю запись…")
        return "break"

    def submit_text(self):
        text = self.text_input.get().strip()
        if not text:
            return
        if not self.engine or not self._acquire_busy():
            self.status.set("Модели загружаются или обрабатывается предыдущий запрос.")
            return
        self.text_input.delete(0, "end")
        self._append("Вы (текст)", text)
        self.send_button.configure(state="disabled")
        self.mic.configure(state="disabled")
        threading.Thread(target=self._run_request, args=(text,), daemon=True, name="request-worker").start()

    def _run_request(self, text):
        try:
            self._ui(self._set_status, "Понимаю запрос и выполняю действие…")
            result = self.engine.handle(text)
            self._ui(self._append, "Ассистент", result)
            self._ui(self._set_status, "Готово. Можно вводить следующий запрос.")
            if self.voice:
                threading.Thread(target=self.voice.speak, args=(result,), daemon=True, name="speech-output").start()
        except Exception as exc:
            self._ui(self._append, "Ошибка", f"{type(exc).__name__}: {exc}")
            self._ui(self._set_status, "Ошибка при обработке запроса. Можно попробовать ещё раз.")
        finally:
            self._ui(self._request_finished)

    def _request_finished(self):
        with self._state_lock:
            self.busy = False
        self.send_button.configure(state="normal" if self.engine else "disabled")
        self.mic.configure(state="normal" if self.engine else "disabled", text="ЗАЖМИТЕ И ГОВОРИТЕ", bg="#d9eaff")

    def close(self):
        self._closed = True
        self.stop_event.set()
        try:
            if self.engine:
                self.engine.repo.close()
        finally:
            self.root.destroy()


def run_app():
    root = tk.Tk()
    App(root)
    root.mainloop()
