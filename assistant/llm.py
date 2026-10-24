import json
import re
import threading

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from .compat import transformers_dtype_kwargs
from .config import LLM_MODEL

SYSTEM_PROMPT = """Ты разбираешь запросы русского голосового помощника для задач, напоминаний и встреч.
Верни ровно один JSON-объект, без Markdown.
Поля: action=create|list|search|delete|reschedule|rename|complete|clarify|unknown;
kind=task|reminder|meeting|any; title=название или поисковая фраза;
date_text=дословное выражение даты/времени из запроса либо пустая строка;
period=today|tomorrow|day_after_tomorrow|week|next_week|specific_date|all или пустая строка; target_id=целое число или null; clarification=вопрос или пустая строка.
Правила: не выдумывай дату, название или ID. «что у меня сегодня/планы на завтра/покажи встречи» — list. Для «послезавтра» используй day_after_tomorrow; для конкретного дня недели или даты — specific_date.
«найди/когда была задача» — search. «удали» — delete; «перенеси/измени дату» — reschedule;
«переименуй/измени название» — rename. Активные дела/задачи — list со статусом open; не включай выполненные записи.
«выполнено/готово» — complete. Создание — create. Не подставляй дату или время по умолчанию. Если при создании задачи, напоминания или встречи не хватает даты или точного времени — action=clarify и задай конкретный вопрос. Время суток вроде «утром», «в обед», «вечером» не является точным временем — спроси часы и минуты.
Если непонятно — unknown. Не выполняй действия, только классифицируй запрос.
"""

DEFAULT_RESULT = {
    "action": "unknown", "kind": "any", "title": "", "date_text": "",
    "period": "", "target_id": None, "clarification": "",
}


class LocalLLM:
    def __init__(self):
        self._generation_lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # Transformers 4.56+ uses `dtype`; avoid the deprecated argument on current versions.
        kwargs = transformers_dtype_kwargs(transformers.__version__, dtype)
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, **kwargs)
        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def _generate(self, messages, max_new_tokens=180):
        with self._generation_lock:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated = output[0, inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def parse(self, user_text, now_iso, records_summary=""):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Текущая дата и время: {now_iso}\nСуществующие записи (только для понимания ссылок):\n{records_summary[:1800]}\nЗапрос: {user_text}\nВерни JSON."},
        ]
        try:
            raw = self._generate(messages)
            match = re.search(r"\{.*\}", raw, re.S)
            if not match:
                return dict(DEFAULT_RESULT)
            data = json.loads(match.group(0))
            result = dict(DEFAULT_RESULT)
            result.update({k: data[k] for k in result.keys() if k in data})
            if result["action"] not in {"create", "list", "search", "delete", "reschedule", "rename", "complete", "clarify", "unknown"}:
                result["action"] = "unknown"
            if result["kind"] not in {"task", "reminder", "meeting", "any"}:
                result["kind"] = "any"
            if result["target_id"] is not None:
                try:
                    result["target_id"] = int(result["target_id"])
                except (TypeError, ValueError):
                    result["target_id"] = None
            return result
        except (ValueError, TypeError, json.JSONDecodeError):
            return dict(DEFAULT_RESULT)
