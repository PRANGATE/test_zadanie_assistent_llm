from datetime import datetime, timedelta
import re

from .database import Repository

KIND_RU = {"task": "задача", "reminder": "напоминание", "meeting": "встреча"}
STATUS_RU = {"open": "активна", "done": "выполнена"}
WEEKDAYS = {
    "понедельник": 0, "понедельника": 0, "понедельнику": 0,
    "вторник": 1, "вторника": 1, "вторнику": 1,
    "среда": 2, "среду": 2, "среды": 2, "среде": 2,
    "четверг": 3, "четверга": 3, "четвергу": 3,
    "пятница": 4, "пятницу": 4, "пятницы": 4, "пятнице": 4,
    "суббота": 5, "субботу": 5, "субботы": 5,
    "воскресенье": 6, "воскресенья": 6, "воскресенью": 6,
}


class AssistantEngine:
    def __init__(self, llm, repo=None, clock=None):
        self.llm = llm
        self.repo = repo or Repository()
        self.clock = clock or datetime.now

    def _summary(self):
        return "\n".join(
            f"ID={r['id']} | {r['kind']} | {r['title']} | {r['due_at'] or 'без даты'} | {r['status']}"
            for r in self.repo.all(30)
        )

    def _parse_date(self, date_text):
        if not date_text:
            return None
        now = self.clock()
        raw_text = str(date_text).strip()

        # Small LLMs sometimes return an ISO timestamp instead of copying the
        # natural-language date. Accept that machine-readable form directly.
        # Examples: 2026-10-21, 2026-10-21T13:57, 2026-10-21T13:57:00.
        iso_candidate = raw_text.replace("Z", "+00:00") if raw_text.endswith("Z") else raw_text
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:[+-]\d{2}:?\d{2})?)?", iso_candidate):
            try:
                parsed_iso = datetime.fromisoformat(iso_candidate)
                if len(iso_candidate) == 10:
                    parsed_iso = parsed_iso.replace(hour=9, minute=0, second=0, microsecond=0)
                else:
                    parsed_iso = parsed_iso.replace(second=0, microsecond=0)
                # Store local wall-clock time consistently with the rest of the app.
                if parsed_iso.tzinfo is not None:
                    parsed_iso = parsed_iso.astimezone().replace(tzinfo=None)
                return parsed_iso.isoformat(timespec="minutes")
            except ValueError:
                return None

        text = raw_text.casefold().replace(",", " ").strip()
        day = None
        if "послезавтра" in text:
            day = (now + timedelta(days=2)).date()
        elif "завтра" in text:
            day = (now + timedelta(days=1)).date()
        elif "сегодня" in text:
            day = now.date()
        else:
            for word, weekday in WEEKDAYS.items():
                if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text):
                    delta = (weekday - now.weekday()) % 7 or 7
                    day = (now + timedelta(days=delta)).date()
                    break

        # Explicit dd.mm[.yyyy] / dd-mm[-yyyy].
        explicit = re.search(r"(?<!\d)(\d{1,2})[.\-/](\d{1,2})(?:[.\-/](\d{2,4}))?(?!\d)", text)
        if day is None and explicit:
            d, m, y = int(explicit.group(1)), int(explicit.group(2)), explicit.group(3)
            year = int(y) if y else now.year
            if year < 100:
                year += 2000
            try:
                day = datetime(year, m, d).date()
                if not y and day < now.date():
                    day = day.replace(year=day.year + 1)
            except ValueError:
                return None

        # Relative duration: in N minutes/hours/days.
        rel = re.search(r"через\s+(\d+)\s+(минут\w*|час\w*|дн\w*)", text)
        if rel:
            n = int(rel.group(1)); unit = rel.group(2)
            if unit.startswith("минут"):
                return (now + timedelta(minutes=n)).replace(second=0, microsecond=0).isoformat(timespec="minutes")
            if unit.startswith("час"):
                return (now + timedelta(hours=n)).replace(second=0, microsecond=0).isoformat(timespec="minutes")
            day = (now + timedelta(days=n)).date()

        clock_match = re.search(r"\bв\s*(\d{1,2}):(\d{2})\b|\b(\d{1,2}):(\d{2})\b|\bв\s*(\d{1,2})(?:\s+(утра|дня|вечера|ночи))?\b|\b(\d{1,2})\s*(утра|дня|вечера|ночи)\b", text)
        hour = minute = None
        if clock_match:
            groups = clock_match.groups()
            if groups[0] is not None:
                h, m, suffix = int(groups[0]), int(groups[1]), None
            elif groups[2] is not None:
                h, m, suffix = int(groups[2]), int(groups[3]), None
            elif groups[4] is not None:
                h, m, suffix = int(groups[4]), 0, groups[5]
            else:
                h, m, suffix = int(groups[6]), 0, groups[7]
            if suffix in {"вечера", "дня"} and 1 <= h <= 11:
                h += 12
            elif suffix == "ночи" and h == 12:
                h = 0
            if h <= 23 and m <= 59:
                hour, minute = h, m

        if day is not None:
            return datetime(day.year, day.month, day.day, hour if hour is not None else 9, minute or 0).isoformat(timespec="minutes")
        if clock_match and any(k in text for k in ("сегодня", "завтра", "послезавтра")):
            return now.replace(hour=hour or 9, minute=minute or 0, second=0, microsecond=0).isoformat(timespec="minutes")
        return None

    def _extract_date_phrase(self, user_text):
        """Conservative date fallback from the original request when the LLM omits it."""
        text = (user_text or "").casefold()
        relative = re.search(r"послезавтра|завтра|сегодня", text)
        if relative:
            phrase = relative.group(0)
            # Keep a nearby explicit time, if present.
            tail = text[relative.start():]
            time_match = re.search(r"\bв\s*\d{1,2}(?::\d{2})?(?:\s+(?:утра|дня|вечера|ночи))?", tail)
            if time_match:
                phrase += " " + time_match.group(0)
            return phrase
        for word in WEEKDAYS:
            match = re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text)
            if match:
                tail = text[match.start():]
                time_match = re.search(r"\bв\s*\d{1,2}(?::\d{2})?(?:\s+(?:утра|дня|вечера|ночи))?", tail)
                return match.group(0) + (" " + time_match.group(0) if time_match else "")
        explicit = re.search(r"(?<!\d)\d{1,2}[.\-/]\d{1,2}(?:[.\-/]\d{2,4})?(?!\d)", text)
        if explicit:
            phrase = explicit.group(0)
            tail = text[explicit.end():]
            time_match = re.search(r"\bв\s*\d{1,2}(?::\d{2})?", tail)
            return phrase + (" " + time_match.group(0) if time_match else "")
        return ""

    def _period_bounds(self, period):
        now = self.clock()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if period == "today":
            start, end = today, today + timedelta(days=1)
        elif period == "tomorrow":
            start, end = today + timedelta(days=1), today + timedelta(days=2)
        elif period == "next_week":
            start = today + timedelta(days=7 - today.weekday())
            end = start + timedelta(days=7)
        else:  # week means rolling seven days starting today
            start, end = today, today + timedelta(days=7)
        return start.isoformat(timespec="minutes"), end.isoformat(timespec="minutes")

    def _format_entry(self, row):
        due = row["due_at"] or "дата не указана"
        return f"#{row['id']} {KIND_RU[row['kind']]}: «{row['title']}», {due}, статус: {STATUS_RU[row['status']]}"

    def handle(self, user_text):
        user_text = (user_text or "").strip()
        if not user_text:
            return "Введите запрос или произнесите его в микрофон."
        parsed = self.llm.parse(user_text, self.clock().isoformat(timespec="minutes"), self._summary())
        if not isinstance(parsed, dict):
            return "Не удалось разобрать запрос. Попробуйте сформулировать иначе."

        # Deterministic safeguards for frequent utterances. Small local models can
        # misclassify a date question as a keyword search or omit the date on create.
        normalized = re.sub(r"\s+", " ", user_text.casefold()).strip(" .!?\t")
        asks_for_period = bool(re.search(
            r"^(?:какая|какие|что|что-нибудь|покажи|список|какие есть|есть ли).{0,70}\b(сегодня|завтра|послезавтра|на неделю|на следующей неделе)\b",
            normalized,
        )) or bool(re.search(r"\b(что у меня|какие у меня|мои задачи|мои встречи).{0,40}\b(сегодня|завтра|послезавтра)\b", normalized))
        if asks_for_period:
            parsed["action"] = "list"
            parsed["period"] = ("tomorrow" if "завтра" in normalized and "послезавтра" not in normalized
                                else "today" if "сегодня" in normalized
                                else "tomorrow" if "послезавтра" in normalized
                                else "next_week" if "следующей неделе" in normalized else "week")
            parsed["title"] = ""
            parsed["date_text"] = ""

        action = parsed.get("action", "unknown")
        kind = parsed.get("kind", "any")
        if kind not in {"task", "reminder", "meeting", "any"}:
            kind = "any"
        title = str(parsed.get("title") or "").strip()
        date_text = str(parsed.get("date_text") or "").strip()
        if action in {"create", "reschedule"} and not date_text:
            date_text = self._extract_date_phrase(user_text)
        due_at = self._parse_date(date_text)

        if action == "clarify":
            return str(parsed.get("clarification") or "Уточните, пожалуйста, дату или время.")
        if action == "unknown":
            return "Не понял запрос. Попробуйте: «добавь задачу…», «что у меня завтра?», «найди задачу…»"
        if action == "create":
            if not title:
                return "Не удалось определить название. Что именно нужно записать?"
            if kind == "any":
                kind = "task"
            if kind == "meeting" and not date_text:
                return f"На какую дату и время поставить встречу «{title}»?"
            if date_text and due_at is None:
                return f"Не смог надёжно разобрать дату «{date_text}». Уточните её, пожалуйста."
            row = self.repo.create(kind, title, due_at)
            return "Создано: " + self._format_entry(row)
        if action == "list":
            period = parsed.get("period", "week")
            if period == "all":
                rows = self.repo.all(100)
            else:
                if period not in {"today", "tomorrow", "week", "next_week"}:
                    period = "week"
                start, end = self._period_bounds(period)
                rows = self.repo.between(start, end, kind)
            if not rows:
                return "За указанный период записей не найдено."
            return "Найдено:\n" + "\n".join(self._format_entry(r) for r in rows[:20])
        if action == "search":
            rows = self.repo.find(title or user_text, kind)
            if not rows:
                return f"Не нашёл запись по запросу «{title or user_text}»."
            return "Нашёл:\n" + "\n".join(self._format_entry(r) for r in rows)
        if action in {"delete", "reschedule", "complete"}:
            target_id = parsed.get("target_id")
            row = self.repo.get(target_id) if isinstance(target_id, int) and not isinstance(target_id, bool) else None
            if row is None:
                matches = self.repo.find(title or user_text, kind)
                if len(matches) > 1 and matches[0]["title"].casefold() == matches[1]["title"].casefold():
                    return "Нашёл несколько похожих записей. Уточните название или ID."
                row = matches[0] if matches else None
            if row is None:
                return "Не нашёл запись. Назовите её точнее."
            if action == "delete":
                self.repo.delete(row["id"])
                return f"Удалено: {KIND_RU[row['kind']]} «{row['title']}» (ID {row['id']})."
            if action == "complete":
                self.repo.complete(row["id"])
                return f"Отмечено выполненным: «{row['title']}»."
            if not date_text or due_at is None:
                return f"На какую дату и время перенести «{row['title']}»?"
            self.repo.update_due(row["id"], due_at)
            return "Перенесено: " + self._format_entry(self.repo.get(row["id"]))
        return "Не удалось выполнить действие. Попробуйте сформулировать запрос иначе."
