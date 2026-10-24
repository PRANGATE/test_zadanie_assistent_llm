from datetime import datetime, timedelta
import re

from .database import Repository
from .semantic_search import SemanticSearch

KIND_RU = {"task": "задача", "reminder": "напоминание", "meeting": "встреча"}
KIND_ACC_RU = {"task": "задачу", "reminder": "напоминание", "meeting": "встречу"}
STATUS_RU = {"open": "активна", "done": "выполнена"}
MONTHS_RU = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря")
def speech_friendly_text(text):
    """Remove database IDs and make the spoken answer less technical."""
    text = re.sub(r"(?<!\w)#\s*\d+\s*", "", str(text))
    if re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", text):
        text = re.sub(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}:\d{2})", r"\3.\2.\1, \4", text)
    return re.sub(r"\s+", " ", text).strip()


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
    def __init__(self, llm, repo=None, clock=None, semantic_search=None):
        self.llm = llm
        self.repo = repo or Repository()
        self.clock = clock or datetime.now
        self.semantic_search = semantic_search or SemanticSearch()
        self.pending_clarification = None

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
                    # Date-only values remain date-only in meaning; midnight is a
                    # parsing sentinel, never a user-facing default appointment time.
                    parsed_iso = parsed_iso.replace(hour=0, minute=0, second=0, microsecond=0)
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

        # Explicit numeric time, including colloquial forms such as "на 15 часов".
        clock_match = re.search(
            r"(?:\bв\s*|\bна\s*)?(\d{1,2}):(\d{2})\b"
            r"|\b(?:в\s*|на\s*)(\d{1,2})(?:\s*(?:час(?:а|ов)?))?(?:\s+(утра|дня|вечера|ночи))?\b"
            r"|\b(\d{1,2})\s*(утра|дня|вечера|ночи)\b", text)
        hour = minute = None
        if clock_match:
            groups = clock_match.groups()
            if groups[0] is not None:
                h, m, suffix = int(groups[0]), int(groups[1]), None
            elif groups[2] is not None:
                h, m, suffix = int(groups[2]), 0, groups[3]
            else:
                h, m, suffix = int(groups[4]), 0, groups[5]
            if suffix in {"вечера", "дня"} and 1 <= h <= 11:
                h += 12
            elif suffix == "ночи" and h == 12:
                h = 0
            if h <= 23 and m <= 59:
                hour, minute = h, m

        if day is not None:
            return datetime(day.year, day.month, day.day, hour if hour is not None else 0, minute or 0).isoformat(timespec="minutes")
        if clock_match and any(k in text for k in ("сегодня", "завтра", "послезавтра")):
            return now.replace(hour=hour if hour is not None else 0, minute=minute or 0, second=0, microsecond=0).isoformat(timespec="minutes")
        return None

    def _extract_date_phrase(self, user_text):
        """Extract date and time cues from the original utterance, independent of LLM output."""
        text = (user_text or "").casefold()
        date_phrase = ""
        relative = re.search(r"послезавтра|завтра|сегодня", text)
        if relative:
            date_phrase = relative.group(0)
        else:
            for word in sorted(WEEKDAYS, key=len, reverse=True):
                match = re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text)
                if match:
                    date_phrase = match.group(0)
                    break
            if not date_phrase:
                explicit = re.search(r"(?<!\d)\d{1,2}[.\-/]\d{1,2}(?:[.\-/]\d{2,4})?(?!\d)", text)
                if explicit:
                    date_phrase = explicit.group(0)
        if not date_phrase:
            return ""

        # Find time anywhere in the request, not just after the date: users often say
        # "на 15 часов, на сегодня". Normalize it to a form the date parser understands.
        numeric = re.search(r"(?:\bв\s*|\bна\s*)(\d{1,2})(?::(\d{2}))?\s*(?:час(?:а|ов)?\b)?(?:\s+(утра|дня|вечера|ночи))?", text)
        if numeric:
            h = int(numeric.group(1)); m = int(numeric.group(2) or 0); suffix = numeric.group(3)
            if suffix in {"дня", "вечера"} and 1 <= h <= 11:
                h += 12
            elif suffix == "ночи" and h == 12:
                h = 0
            if h <= 23 and m <= 59:
                return f"{date_phrase} в {h:02d}:{m:02d}"
        # Vague dayparts are not silently converted into a clock time. Ask the
        # user for an exact time instead.
        return date_phrase


    def _extract_title_from_request(self, user_text):
        """Extract the task payload from common Russian create commands."""
        text = re.sub(r"\s+", " ", (user_text or "").strip()).strip(" .!?\t")
        text = re.sub(
            r"^(?:пожалуйста\s+)?(?:добавь|добавить|создай|создать|запиши|записать|поставь|поставить|напомни|напомнить)\s+",
            "", text, flags=re.IGNORECASE,
        )
        text = re.sub(r"^(?:мне\s+)?(?:задачу|задача|задачку|напоминание|напоминалку)\b\s*", "", text, flags=re.IGNORECASE)
        # Remove temporal phrases from the title, but leave the actual action.
        text = re.sub(r"\bна\s+(?:послезавтра|завтра|сегодня)\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:послезавтра|завтра|сегодня)\b", " ", text, flags=re.IGNORECASE)
        for weekday in sorted(WEEKDAYS, key=len, reverse=True):
            text = re.sub(rf"\b(?:на\s+)?{re.escape(weekday)}\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:на\s+)?\d{1,2}[.\-/]\d{1,2}(?:[.\-/]\d{2,4})?\b", " ", text)
        text = re.sub(r"\b(?:в|на)\s*\d{1,2}(?::\d{2})?(?:\s*час(?:а|ов)?)?(?:\s+(?:утра|дня|вечера|ночи))?\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bв\s+обед\b|\bобед(?:ом)?\b|\bутром\b|\bвечером\b|\bночью\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" -—,:;.!?\t")
        # Preserve a meaningful meeting title after stripping the command verb.
        if re.match(r"^встречу\b", text, flags=re.IGNORECASE):
            text = re.sub(r"^встречу\b", "Встреча", text, count=1, flags=re.IGNORECASE)
        # Do not replace a potentially useful model title with a tiny/noisy fragment.
        if len(text) >= 3:
            return text[0].upper() + text[1:]
        return ""

    def _extract_explicit_time(self, text):
        """Return an explicit clock time from a phrase, or None."""
        text = (text or "").casefold()
        match = re.search(r"(?:\bв\s*|\bна\s*)?(\d{1,2})(?::(\d{2}))?\s*(?:час(?:а|ов)?\b)?(?:\s+(утра|дня|вечера|ночи))?", text)
        if match:
            h = int(match.group(1)); minute = int(match.group(2) or 0); suffix = match.group(3)
            # Reject incidental small numbers unless they are explicitly introduced or have a suffix.
            introduced = re.search(r"\b(?:в|на)\s*\d", text) or ":" in match.group(0) or "час" in match.group(0) or suffix
            if introduced:
                if suffix in {"дня", "вечера"} and 1 <= h <= 11:
                    h += 12
                elif suffix == "ночи" and h == 12:
                    h = 0
                if 0 <= h <= 23 and 0 <= minute <= 59:
                    return h, minute
        return None

    def _finish_pending_clarification(self, user_text):
        pending = self.pending_clarification
        if not pending:
            return None
        normalized = user_text.casefold().strip(" .!?")
        if normalized in {"отмена", "отмени", "не надо", "нет", "забудь", "отменить"}:
            self.pending_clarification = None
            return "Хорошо, создание записи отменено."

        kind = pending.get("kind", "task")
        title = pending["title"]
        date_text = pending.get("date_text", "")
        time_value = pending.get("time_value")
        supplied_date = self._extract_date_phrase(user_text)
        supplied_time = self._extract_explicit_time(user_text)

        if not date_text and supplied_date:
            date_text = supplied_date
        if supplied_time:
            time_value = supplied_time

        parsed_date = self._parse_date(date_text) if date_text else None
        if not parsed_date:
            self.pending_clarification = {"type": "create_date", "kind": kind, "title": title, "date_text": date_text, "time_value": time_value}
            self.pending_clarification = {"type": "create_date", "kind": kind, "title": title, "date_text": "", "time_value": time_value}
            return f"На какую дату поставить {KIND_ACC_RU[kind]} «{title}»? Укажите дату, например: завтра или 25.10.2026."

        if not time_value:
            self.pending_clarification = {"type": "create_time", "kind": kind, "title": title, "date_text": date_text}
            date_label = self._format_date_only(parsed_date)
            return f"Дата: {date_label}. На какое точное время поставить {KIND_ACC_RU[kind]} «{title}»?"

        base = datetime.fromisoformat(parsed_date)
        due_at = base.replace(hour=time_value[0], minute=time_value[1], second=0, microsecond=0).isoformat(timespec="minutes")
        row = self.repo.create(kind, title, due_at)
        self.pending_clarification = None
        return "Создано: " + self._format_entry(row)

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

    def _format_date_only(self, value):
        if not value:
            return "дата не указана"
        try:
            dt = datetime.fromisoformat(str(value))
            return f"{dt.day} {MONTHS_RU[dt.month - 1]} {dt.year} года"
        except (TypeError, ValueError):
            return str(value)

    def _format_due(self, value):
        if not value:
            return "дата не указана"
        try:
            dt = datetime.fromisoformat(str(value))
            return f"{dt.day} {MONTHS_RU[dt.month - 1]} {dt.year} года, {dt:%H:%M}"
        except (TypeError, ValueError):
            return str(value)

    def _format_entry(self, row):
        due = self._format_due(row["due_at"])
        return f"#{row['id']} {KIND_RU[row['kind']]}: «{row['title']}», срок: {due}, статус: {STATUS_RU[row['status']]}"

    def handle(self, user_text):
        user_text = (user_text or "").strip()
        if not user_text:
            return "Введите запрос или произнесите его в микрофон."
        if self.pending_clarification:
            return self._finish_pending_clarification(user_text)
        parsed = self.llm.parse(user_text, self.clock().isoformat(timespec="minutes"), self._summary())
        if not isinstance(parsed, dict):
            return "Не удалось разобрать запрос. Попробуйте сформулировать иначе."

        # Deterministic safeguards for frequent utterances. Small local models can
        # misclassify a date question as a keyword search or omit the date on create.
        normalized = re.sub(r"\s+", " ", user_text.casefold()).strip(" .!?\t")
        asks_for_period = bool(re.search(
            r"^(?:(?:какая|какие|что|что-нибудь|покажи|список|какие есть|есть ли).{0,70}|(?:задач\w*|встреч\w*|напоминани\w*).{0,35})\b(сегодня|завтра|послезавтра|на неделю|на следующей неделе)\b",
            normalized,
        )) or bool(re.search(r"\b(что у меня|какие у меня|мои задачи|мои встречи).{0,40}\b(сегодня|завтра|послезавтра)\b", normalized))
        is_create_utterance = bool(re.match(r"^(?:пожалуйста\s+)?(?:добавь|добавить|создай|создать|запиши|записать|поставь|поставить|напомни|напомнить|назначь|запланируй)\b", normalized))
        if asks_for_period and not is_create_utterance:
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
        source_date = self._extract_date_phrase(user_text)
        # Do not let the small LLM's generic "clarify" classification bypass the
        # actual multi-turn meeting flow. Detect explicit meeting-creation wording.
        meeting_create = bool(re.search(
            r"\b(?:добавь|добавить|создай|создать|поставь|поставить|запланируй|назначь)\b.{0,50}\bвстреч\w*\b",
            normalized,
        ))
        if meeting_create:
            action = "create"
            kind = "meeting"
            parsed["action"] = action
            parsed["kind"] = kind
            extracted_meeting_title = self._extract_title_from_request(user_text)
            if extracted_meeting_title:
                title = extracted_meeting_title
            if source_date:
                date_text = source_date
        if action == "create":
            # The original utterance is the source of truth for relative dates;
            # small LLMs can hallucinate yesterday's date or return midnight.
            if source_date and re.search(r"\b(?:сегодня|завтра|послезавтра)\b", source_date):
                date_text = source_date
            source_title = self._extract_title_from_request(user_text)
            if source_title:
                title = source_title
            else:
                # Clean common instruction/date debris from a model-produced title.
                title = re.sub(r"^(?:добавь|добавить|создай|создать)\s+", "", title, flags=re.IGNORECASE)
                title = re.sub(r"^(?:мне\s+)?(?:задачу|задача|задачку)\s*", "", title, flags=re.IGNORECASE)
                title = re.sub(r"\bна\s+(?:послезавтра|завтра|сегодня)\b", " ", title, flags=re.IGNORECASE).strip(" -—,:;.!?\t")
        elif action == "reschedule" and not date_text:
            date_text = source_date
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

            # Never invent a date or time. Every new task/reminder/meeting must
            # have both explicitly supplied by the user before it is persisted.
            explicit_time = self._extract_explicit_time(user_text)
            if not date_text and source_date:
                date_text = source_date
            if not date_text and due_at:
                date_text = due_at
            parsed_day = self._parse_date(date_text) if date_text else None
            if date_text and not parsed_day:
                return f"Не смог надёжно разобрать дату «{date_text}». Уточните её, пожалуйста."
            if not parsed_day:
                self.pending_clarification = {"type": "create_date", "kind": kind, "title": title, "date_text": "", "time_value": explicit_time}
                return f"На какую дату поставить {KIND_ACC_RU[kind]} «{title}»? Укажите дату, например: завтра или 25.10.2026."
            if not explicit_time:
                self.pending_clarification = {"type": "create_time", "kind": kind, "title": title, "date_text": date_text}
                return f"Дата: {self._format_date_only(parsed_day)}. На какое точное время поставить {KIND_ACC_RU[kind]} «{title}»?"
            base = datetime.fromisoformat(parsed_day)
            due_at = base.replace(hour=explicit_time[0], minute=explicit_time[1], second=0, microsecond=0).isoformat(timespec="minutes")
            row = self.repo.create(kind, title, due_at)
            return "Создано: " + self._format_entry(row)
        if action == "list":
            period = parsed.get("period")
            if not period:
                return "За какой период показать записи: сегодня, завтра, на неделю или все записи?"
            if period == "all":
                rows = self.repo.all(100)
            else:
                if period not in {"today", "tomorrow", "week", "next_week"}:
                    return "За какой период показать записи: сегодня, завтра, на неделю или все записи?"
                start, end = self._period_bounds(period)
                rows = self.repo.between(start, end, kind)
            if not rows:
                return "За указанный период записей не найдено."
            return "Найдено:\n" + "\n".join(self._format_entry(r) for r in rows[:20])
        if action == "search":
            query = title or user_text
            rows = self.repo.find(query, kind)
            # If lexical matching is weak or empty, try local multilingual embeddings.
            if len(rows) < 3:
                candidates = [r for r in self.repo.all(1000) if kind in {"any", r["kind"]}]
                try:
                    semantic_rows = self.semantic_search.rank(user_text, candidates, limit=10)
                    known = {r["id"] for r in rows}
                    rows.extend(r for r in semantic_rows if r["id"] not in known)
                except Exception:
                    # Semantic model is optional until its first successful local download.
                    pass
            if not rows:
                return f"Не нашёл запись по запросу «{query}»."
            return "Нашёл:\n" + "\n".join(self._format_entry(r) for r in rows[:10])
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
