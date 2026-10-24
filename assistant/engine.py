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

RUSSIAN_NUMBERS = {
    "ноль": 0, "один": 1, "одна": 1, "одно": 1, "одну": 1,
    "одного": 1, "одной": 1, "одном": 1, "одному": 1,
    "два": 2, "две": 2, "двух": 2, "двум": 2,
    "три": 3, "трёх": 3, "трех": 3, "трем": 3, "трём": 3,
    "четыре": 4, "четырёх": 4, "четырех": 4,
    "пять": 5, "пяти": 5, "шесть": 6, "шести": 6,
    "семь": 7, "семи": 7, "восемь": 8, "восьми": 8,
    "девять": 9, "девяти": 9, "десять": 10, "десяти": 10,
    "одиннадцать": 11, "одиннадцати": 11,
    "двенадцать": 12, "двенадцати": 12,
    "тринадцать": 13, "тринадцати": 13,
    "четырнадцать": 14, "четырнадцати": 14,
    "пятнадцать": 15, "пятнадцати": 15,
    "шестнадцать": 16, "шестнадцати": 16,
    "семнадцать": 17, "семнадцати": 17,
    "восемнадцать": 18, "восемнадцати": 18,
    "девятнадцать": 19, "девятнадцати": 19,
    "двадцать": 20, "двадцати": 20,
    "двадцать один": 21, "двадцать одна": 21,
    "двадцать два": 22, "двадцать две": 22,
    "двадцать три": 23, "двадцать четыре": 24,
}
MONTHS_RU_PARSE = {
    "января": 1, "январе": 1, "февраля": 2, "феврале": 2,
    "марта": 3, "марте": 3, "апреля": 4, "апреле": 4,
    "мая": 5, "июня": 6, "июне": 6, "июля": 7, "июле": 7,
    "августа": 8, "августе": 8, "сентября": 9, "сентябре": 9,
    "октября": 10, "октябре": 10, "ноября": 11, "ноябре": 11,
    "декабря": 12, "декабре": 12,
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

    @staticmethod
    def _duration_match(text):
        text = (text or "").casefold().replace("ё", "ё")
        if re.search(r"\bчерез\s+полчаса\b", text):
            return ("minutes", 30, re.search(r"\bчерез\s+полчаса\b", text).group(0))
        number_words = "|".join(sorted((re.escape(k) for k in RUSSIAN_NUMBERS), key=len, reverse=True))
        match = re.search(
            rf"\bчерез\s+(?P<n>\d+|{number_words})\s+(?P<unit>минут\w*|час\w*|дн\w*)\b",
            text, re.IGNORECASE,
        )
        if not match:
            return None
        raw_n = match.group("n").casefold()
        count = int(raw_n) if raw_n.isdigit() else RUSSIAN_NUMBERS.get(raw_n)
        if count is None:
            return None
        unit_raw = match.group("unit").casefold()
        unit = "minutes" if unit_raw.startswith("минут") else "hours" if unit_raw.startswith("час") else "days"
        return unit, count, match.group(0)

    def _parse_date(self, date_text):
        """Parse an explicit date phrase; midnight is a date-only sentinel, never a default appointment time."""
        if not date_text:
            return None
        now = self.clock()
        raw_text = str(date_text).strip()

        # Accept machine-readable timestamps only when the phrase itself is an ISO value.
        iso_candidate = raw_text.replace("Z", "+00:00") if raw_text.endswith("Z") else raw_text
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:[+-]\d{2}:?\d{2})?)?", iso_candidate):
            try:
                parsed_iso = datetime.fromisoformat(iso_candidate)
                if len(iso_candidate) == 10:
                    parsed_iso = parsed_iso.replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    parsed_iso = parsed_iso.replace(second=0, microsecond=0)
                if parsed_iso.tzinfo is not None:
                    # Preserve the wall-clock value supplied in the user's text. Do not
                    # reinterpret it using the machine timezone of the host environment.
                    parsed_iso = parsed_iso.replace(tzinfo=None)
                return parsed_iso.isoformat(timespec="minutes")
            except ValueError:
                return None

        text = raw_text.casefold().replace(",", " ").strip()
        day = None
        duration = self._duration_match(text)
        if duration:
            unit, count, _phrase = duration
            if unit == "minutes":
                return (now + timedelta(minutes=count)).replace(second=0, microsecond=0).isoformat(timespec="minutes")
            if unit == "hours":
                return (now + timedelta(hours=count)).replace(second=0, microsecond=0).isoformat(timespec="minutes")
            day = (now + timedelta(days=count)).date()
        elif re.search(r"\bпослезавтра\b", text):
            day = (now + timedelta(days=2)).date()
        elif re.search(r"\bзавтра\b", text):
            day = (now + timedelta(days=1)).date()
        elif re.search(r"\bсегодня\b", text):
            day = now.date()
        elif re.search(r"\bдо\s+конца\s+недели\b", text):
            day = (now + timedelta(days=(6 - now.weekday()))).date()
        else:
            for word, weekday in sorted(WEEKDAYS.items(), key=lambda x: len(x[0]), reverse=True):
                if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text):
                    delta = (weekday - now.weekday()) % 7
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

        # Russian written month dates: "25 октября" / "25 октября 2026".
        if day is None:
            month_pattern = "|".join(sorted((re.escape(m) for m in MONTHS_RU_PARSE), key=len, reverse=True))
            verbal = re.search(rf"(?<!\d)(\d{{1,2}})\s+({month_pattern})(?:\s+(\d{{4}}))?", text)
            if verbal:
                d = int(verbal.group(1)); m = MONTHS_RU_PARSE[verbal.group(2)]
                year = int(verbal.group(3) or now.year)
                try:
                    day = datetime(year, m, d).date()
                    if not verbal.group(3) and day < now.date():
                        day = day.replace(year=day.year + 1)
                except ValueError:
                    return None

        # Extract only a clearly introduced/colon-formatted clock time. Strip date numerals
        # first, so a date like 25.10.2026 can never be misread as 25:00.
        clock_source = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2})?\b", " ", text)
        clock_source = re.sub(r"(?<!\d)\d{1,2}[.\-/]\d{1,2}(?:[.\-/]\d{2,4})?(?!\d)", " ", clock_source)
        time_value = self._extract_explicit_time(clock_source)
        if day is not None:
            hour, minute = time_value if time_value is not None else (0, 0)
            return datetime(day.year, day.month, day.day, hour, minute).isoformat(timespec="minutes")

        # A time without a date is intentionally not converted into today's date.
        return None

    def _extract_date_phrase(self, user_text):
        """Extract date/duration facts from the user's words, never from model guesses."""
        text = (user_text or "").casefold()

        duration = self._duration_match(text)
        if duration and duration[0] in {"minutes", "hours"}:
            return duration[2]
        if re.search(r"\bпослезавтра\b", text):
            date_phrase = "послезавтра"
        elif re.search(r"\bзавтра\b", text):
            date_phrase = "завтра"
        elif re.search(r"\bсегодня\b", text):
            date_phrase = "сегодня"
        elif re.search(r"\bдо\s+конца\s+недели\b", text):
            date_phrase = "до конца недели"
        else:
            date_phrase = ""
            for word in sorted(WEEKDAYS, key=len, reverse=True):
                match = re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text)
                if match:
                    date_phrase = match.group(0)
                    break
            if not date_phrase:
                iso = re.search(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2})?\b", text)
                explicit = re.search(r"(?<!\d)\d{1,2}[.\-/]\d{1,2}(?:[.\-/]\d{2,4})?(?!\d)", text)
                month_pattern = "|".join(sorted((re.escape(m) for m in MONTHS_RU_PARSE), key=len, reverse=True))
                verbal = re.search(rf"(?<!\d)\d{{1,2}}\s+(?:{month_pattern})(?:\s+\d{{4}})?", text)
                date_match = iso or explicit or verbal
                if date_match:
                    date_phrase = date_match.group(0)

        if not date_phrase:
            if duration and duration[0] == "days":
                return duration[2]
            return ""

        # Capture an explicitly stated time anywhere in the utterance, but not a number
        # belonging to a calendar date. Relative hour/minute durations already include time.
        time_value = self._extract_explicit_time(text)
        if time_value is not None:
            return f"{date_phrase} в {time_value[0]:02d}:{time_value[1]:02d}"
        return date_phrase

    def _extract_title_from_request(self, user_text):
        """Extract a human task title from common Russian commands/utterances."""
        text = re.sub(r"\s+", " ", (user_text or "").strip()).strip(" .!?\t")
        text = re.sub(
            r"^(?:пожалуйста\s+)?(?:добавь|добавить|создай|создать|запиши|записать|поставь|поставить|напомни|напомнить|назначь|назначить|запланируй|запланировать)\s+",
            "", text, flags=re.IGNORECASE,
        )
        text = re.sub(r"^(?:(?:мне|нам)\s+)?(?:задачу|задача|задачку|напоминание|напоминалку)\b\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:мне\s+)?", "", text, flags=re.IGNORECASE)
        # Remove explicit and relative date phrases from the title.
        temporal_patterns = [
            r"\bчерез\s+(?:\d+|одиннадцать|двенадцать|тринадцать|четырнадцать|пятнадцать|шестнадцать|семнадцать|восемнадцать|девятнадцать|двадцать(?:\s+(?:один|одна|два|две|три|четыре))?|один|одна|одно|одну|две|два|двух|три|трёх|трех|четыре|четырёх|четырех|пять|пяти|шесть|шести|семь|семи|восемь|восьми|девять|девяти|десять|десяти)\s+(?:минут\w*|час\w*|дн\w*)",
            r"\bчерез\s+полчаса\b", r"\bдо\s+конца\s+недели\b",
            r"\bна\s+(?:послезавтра|завтра|сегодня)\b", r"\b(?:послезавтра|завтра|сегодня)\b",
            r"\b(?:на\s+)?(?:следующей\s+неделе|этой\s+неделе)\b",
            r"\b(?:на\s+)?\d{1,2}[.\-/]\d{1,2}(?:[.\-/]\d{2,4})?\b",
            r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2})?\b",
            r"\b(?:в|на)\s*\d{1,2}(?::\d{2})?(?:\s*час(?:а|ов)?)?(?:\s+(?:утра|дня|вечера|ночи))?\b",
            r"\b\d{1,2}\s+(?:утра|дня|вечера|ночи)\b",
            r"\bв\s+обед\b|\bобед(?:ом)?\b|\bутром\b|\bвечером\b|\bночью\b",
        ]
        for pattern in temporal_patterns:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        for weekday in sorted(WEEKDAYS, key=len, reverse=True):
            text = re.sub(rf"\b(?:на\s+)?{re.escape(weekday)}\b", " ", text, flags=re.IGNORECASE)
        month_pattern = "|".join(sorted((re.escape(m) for m in MONTHS_RU_PARSE), key=len, reverse=True))
        text = re.sub(rf"\b\d{{1,2}}\s+(?:{month_pattern})(?:\s+\d{{4}})?\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" -—,:;.!?\t")
        text = re.sub(r"^(?:в|на)\s+(?=(?:(?:мне|нам|у меня|у нас)\s+)?(?:нужно|надо|необходимо|пора|у меня|у нас|встреча|отправить|позвонить|купить|сделать)\b)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:(?:мне|нам|у меня|у нас)\s+)?(?:нужно|надо|необходимо|пора)\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:у меня|у нас)\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:встреча|встречу)\s+", "Встреча ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" -—,:;.!?\t")
        if len(text) >= 3:
            return text[0].upper() + text[1:]
        return ""

    def _extract_explicit_time(self, text):
        """Return a time explicitly said by the user, including a relative hour/minute duration."""
        text = (text or "").casefold()
        duration = self._duration_match(text)
        if duration and duration[0] in {"minutes", "hours"}:
            delta = timedelta(minutes=duration[1]) if duration[0] == "minutes" else timedelta(hours=duration[1])
            due = (self.clock() + delta).replace(second=0, microsecond=0)
            return due.hour, due.minute
        if duration and duration[0] == "days":
            text = text.replace(duration[2], " ", 1)

        # Remove date tokens so the day/month in a date cannot accidentally become a time.
        cleaned = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2})?\b", " ", text)
        cleaned = re.sub(r"(?<!\d)\d{1,2}[.\-/]\d{1,2}(?:[.\-/]\d{2,4})?(?!\d)", " ", cleaned)
        patterns = [
            r"\b(?:в|на)\s*(\d{1,2}):(\d{2})\b",
            r"\b(?:в|на)\s*(\d{1,2})(?:\s*час(?:а|ов)?\b)?(?:\s+(утра|дня|вечера|ночи))?",
            r"\b(\d{1,2}):(\d{2})\b",
            r"\b(\d{1,2})\s+(утра|дня|вечера|ночи)\b",
        ]
        for index, pattern in enumerate(patterns):
            match = re.search(pattern, cleaned)
            if not match:
                continue
            if index in (0, 2):
                hour, minute = int(match.group(1)), int(match.group(2))
                suffix = None
            else:
                hour = int(match.group(1)); minute = 0
                suffix = match.group(2) if index == 1 else match.group(2)
            if suffix in {"дня", "вечера"} and 1 <= hour <= 11:
                hour += 12
            elif suffix == "ночи" and hour == 12:
                hour = 0
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return hour, minute
        return None

    def _finish_pending_clarification(self, user_text):
        pending = self.pending_clarification
        if not pending:
            return None
        normalized = user_text.casefold().strip(" .!?")
        if normalized in {"отмена", "отмени", "не надо", "нет", "забудь", "отменить"}:
            self.pending_clarification = None
            return "Хорошо, создание записи отменено."

        if pending.get("type") == "rename_title":
            new_title = user_text.strip().strip('«»\"\' ').strip(" .!?")
            if not new_title:
                return "Введите новое название записи."
            if len(new_title) > 200:
                return "Название слишком длинное. Укажите не более 200 символов."
            updated = self.repo.update_title(pending["entry_id"], new_title)
            row = self.repo.get(pending["entry_id"]) if updated else None
            self.pending_clarification = None
            if not row:
                return "Не удалось изменить название: запись не найдена."
            return "Название изменено: " + self._format_entry(row)

        if pending.get("type") == "edit_field":
            if re.search(r"\b(?:названи\w*|имя|переименован\w*)\b", normalized):
                self.pending_clarification = {
                    "type": "rename_title", "entry_id": pending["entry_id"],
                    "title": pending["title"], "kind": pending["kind"],
                }
                return f"Какое новое название задать записи «{pending['title']}»?"
            if re.search(r"\b(?:дат\w*|врем\w*|срок\w*|перенос\w*)\b", normalized):
                self.pending_clarification = {
                    "type": "reschedule_date", "entry_id": pending["entry_id"],
                    "title": pending["title"], "kind": pending["kind"],
                    "date_text": "", "time_value": None,
                }
                return f"Укажите новую дату и точное время для «{pending['title']}», например: завтра в 15:30."
            return "Что изменить: название или срок (дату и время)?"

        if pending.get("type") == "create_kind":
            answer = user_text.casefold()
            if re.search(r"\b(?:задач\w*|дело)\b", answer):
                resolved_kind = "task"
            elif re.search(r"\b(?:напоминани\w*|напоминалк\w*|напомни)\b", answer):
                resolved_kind = "reminder"
            elif re.search(r"\b(?:встреч\w*|созвон)\b", answer):
                resolved_kind = "meeting"
            else:
                return "Уточните тип записи: задача, напоминание или встреча?"
            self.pending_clarification = {
                "type": "create_date", "kind": resolved_kind, "title": pending["title"],
                "date_text": pending.get("date_text", ""), "time_value": pending.get("time_value"),
            }
            return self._finish_pending_clarification(user_text)

        if pending.get("type", "").startswith("reschedule"):
            date_text = pending.get("date_text", "")
            time_value = pending.get("time_value")
            supplied_date = self._extract_date_phrase(user_text)
            supplied_time = self._extract_explicit_time(user_text)
            if supplied_date and not date_text:
                date_text = supplied_date
            if supplied_time is not None:
                time_value = supplied_time
            parsed_date = self._parse_date(date_text) if date_text else None
            # A time-only response must not cause the assistant to ask for the date forever.
            # Preserve the current date when the selected entry already has one.
            existing = self.repo.get(pending["entry_id"])
            if not parsed_date and not date_text and time_value is not None and existing and existing.get("due_at"):
                try:
                    existing_dt = datetime.fromisoformat(existing["due_at"])
                    due_at = existing_dt.replace(hour=time_value[0], minute=time_value[1], second=0, microsecond=0).isoformat(timespec="minutes")
                except (TypeError, ValueError):
                    due_at = None
                if due_at:
                    updated = self.repo.update_due(pending["entry_id"], due_at)
                    self.pending_clarification = None
                    if not updated:
                        return "Не удалось найти запись для переноса: возможно, её удалили."
                    return "Перенесено: " + self._format_entry(self.repo.get(pending["entry_id"]))
            if not parsed_date:
                self.pending_clarification = {
                    "type": "reschedule_date", "entry_id": pending["entry_id"],
                    "title": pending["title"], "kind": pending["kind"],
                    "date_text": "", "time_value": time_value,
                }
                return f"На какую дату перенести «{pending['title']}»?"
            if time_value is None:
                self.pending_clarification = {
                    "type": "reschedule_time", "entry_id": pending["entry_id"],
                    "title": pending["title"], "kind": pending["kind"],
                    "date_text": date_text, "time_value": None,
                }
                return f"Дата: {self._format_date_only(parsed_date)}. На какое точное время перенести «{pending['title']}»?"
            base = datetime.fromisoformat(parsed_date)
            due_at = base.replace(hour=time_value[0], minute=time_value[1], second=0, microsecond=0).isoformat(timespec="minutes")
            updated = self.repo.update_due(pending["entry_id"], due_at)
            self.pending_clarification = None
            if not updated:
                return "Не удалось найти запись для переноса: возможно, её удалили."
            return "Перенесено: " + self._format_entry(self.repo.get(pending["entry_id"]))

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
        elif period == "day_after_tomorrow":
            start, end = today + timedelta(days=2), today + timedelta(days=3)
        elif period == "next_week":
            start = today + timedelta(days=7 - today.weekday())
            end = start + timedelta(days=7)
        else:  # current calendar week, Monday through next Monday
            start = today - timedelta(days=today.weekday())
            end = start + timedelta(days=7)
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

    def _resolve_mutation_target(self, user_text, parsed_title="", kind="any"):
        """Resolve a mutation target conservatively; never mutate a fuzzy guess silently.

        Returns (row, candidates, explicit_id). ``explicit_id`` is set even if the ID
        does not exist, which prevents fallback to a different row.
        """
        id_match = re.search(r"(?:#\s*|\b(?:id|номер|запись)\s*#?\s*)(\d+)\b", user_text, re.IGNORECASE)
        if id_match:
            explicit_id = int(id_match.group(1))
            return self.repo.get(explicit_id), [], explicit_id

        normalized = (user_text or "").casefold()
        all_candidates = self.repo.all(1000)
        # Prefer a record whose complete stored title occurs verbatim in the command,
        # even if the LLM guessed the wrong entry kind.
        literal_direct = [r for r in all_candidates if len(r["title"].strip()) >= 3 and r["title"].casefold() in normalized]
        if literal_direct:
            longest = max(len(r["title"]) for r in literal_direct)
            literal_direct = [r for r in literal_direct if len(r["title"]) == longest]
            if len(literal_direct) == 1:
                return literal_direct[0], [], None
            return None, literal_direct, None

        # Explicit words like "встречу" are stronger evidence than a mistaken LLM
        # kind label. This matters when a task and a meeting contain similar words.
        search_kind = kind
        if re.search(r"\b(?:встреч\w*|созвон\w*)\b", normalized):
            search_kind = "meeting"
        elif re.search(r"\b(?:напоминани\w*|напоминалк\w*)\b", normalized):
            search_kind = "reminder"
        candidates = [r for r in all_candidates if search_kind in {"any", r["kind"]}]
        direct = [r for r in candidates if len(r["title"].strip()) >= 3 and r["title"].casefold() in normalized]
        if direct:
            longest = max(len(r["title"]) for r in direct)
            direct = [r for r in direct if len(r["title"]) == longest]
            if len(direct) == 1:
                return direct[0], [], None
            return None, direct, None

        query = (parsed_title or "").strip()
        if query:
            exact = [r for r in candidates if r["title"].casefold().strip() == query.casefold()]
            if len(exact) == 1:
                return exact[0], [], None
            if len(exact) > 1:
                return None, exact, None

        # Search both the title predicted by the LLM and the full original command.
        # The small LLM may return a grammatical variant of the real title or assign
        # the wrong kind. If there is no match of that kind, retry across all kinds.
        # Ambiguous results are still returned to the caller for clarification.
        queries = []
        for candidate_query in (query, user_text):
            candidate_query = (candidate_query or "").strip()
            if candidate_query and candidate_query.casefold() not in {q.casefold() for q in queries}:
                queries.append(candidate_query)
        matches_by_id = {}
        for candidate_query in queries:
            for item in self.repo.find(candidate_query, search_kind, limit=10):
                matches_by_id[item["id"]] = item
        matches = list(matches_by_id.values())
        if not matches and kind != "any":
            for candidate_query in queries:
                for item in self.repo.find(candidate_query, "any", limit=10):
                    matches_by_id[item["id"]] = item
            matches = list(matches_by_id.values())
        if len(matches) == 1 and query:
            return matches[0], [], None
        return None, matches, None

    def _extract_new_title(self, user_text, row):
        quoted = re.findall(r"[«“\"]([^»”\"]+)[»”\"]", user_text or "")
        if len(quoted) >= 2:
            return quoted[-1].strip()
        id_match = re.search(r"(?:#\s*|\b(?:id|номер|запись)\s*#?\s*)(\d+)\b", user_text or "", re.IGNORECASE)
        if len(quoted) == 1 and id_match:
            return quoted[0].strip()
        source = user_text or ""
        index = source.casefold().find(row["title"].casefold())
        if index >= 0:
            tail = source[index + len(row["title"]):].lstrip('»”"\' \t')
            tail = re.sub(r"^\s*(?:на|вместо|как|в|будет)\s+", "", tail, flags=re.IGNORECASE)
            tail = re.sub(r"^\s*(?:новое\s+)?названи\w*\s*[:=]?\s*", "", tail, flags=re.IGNORECASE)
            tail = tail.strip().strip('«»“”\"\' ').strip(" .!?,;:")
            # A transition word is required: don't treat trailing politeness words as a title.
            if tail and not re.fullmatch(r"(?:пожалуйста|название|имя|задачу|задача)", tail, re.IGNORECASE):
                return tail
        if id_match:
            tail = source[id_match.end():]
            tail = re.sub(r"^\s*(?:задач\w*|дело|дела|запись|напоминани\w*|встреч\w*)\s*", "", tail, flags=re.IGNORECASE)
            tail = re.sub(r"^\s*(?:на|вместо|как|в|будет)\s+", "", tail, flags=re.IGNORECASE)
            tail = tail.strip().strip('«»“”\"\' ').strip(" .!?,;:")
            if tail:
                return tail
        return ""

    def handle(self, user_text):
        user_text = (user_text or "").strip()
        if not user_text:
            return "Введите запрос или произнесите его в микрофон."
        if self.pending_clarification:
            return self._finish_pending_clarification(user_text)
        parsed = self.llm.parse(user_text, self.clock().isoformat(timespec="minutes"), self._summary())
        if not isinstance(parsed, dict):
            return "Не удалось разобрать запрос. Попробуйте сформулировать иначе."

        # Deterministic safeguards: natural-language facts from the user are authoritative;
        # a small LLM may classify an action, but cannot invent a deadline/time.
        normalized = re.sub(r"\s+", " ", user_text.casefold()).strip(" .!?\t")
        normalized = normalized.replace("ё", "е")
        period_match = re.search(
            r"\b(послезавтра|завтра|сегодня|на\s+следующей\s+неделе|на\s+этой\s+неделе|на\s+эту\s+неделю|эту\s+неделю|этой\s+неделе|на\s+неделю)\b",
            normalized,
        )
        list_lead = bool(re.search(
            r"^(?:вывод\w*|выведи|вывести|покажи|показать|список|какая?\s+задач\w*|какие\s+задач\w*|задач\w*|встреч\w*|событи\w*|напоминани\w*|что\b|что\s+запланировано|(?:какие\s+)?планы\b|есть\s+ли|мои\s+(?:задач\w*|встреч\w*|событи\w*|напоминани\w*)|активн\w*|незаверш[еёе]н\w*|открыт\w*)",
            normalized,
        ))
        asks_for_period = bool(period_match and list_lead)
        is_create_utterance = bool(re.match(
            r"^(?:пожалуйста\s+)?(?:добавь|добавить|создай|создать|запиши|записать|поставь|поставить|напомни|напомнить|назначь|назначить|запланируй|запланировать)\b",
            normalized,
        ))
        active_query = bool(re.search(r"\b(?:активн\w*|незаверш[еёе]н\w*|открыт\w*)\b", normalized)) and not is_create_utterance
        natural_task_create = bool(
            re.search(r"\b(?:нужно|надо|необходимо|пора)\b", normalized)
            and (self._extract_date_phrase(user_text) or re.search(r"\b(?:задач\w*|отч[её]т|позвонить|отправить|купить|сделать)\b", normalized))
        )
        natural_meeting_create = bool(re.search(
            r"\b(?:у\s+меня|у\s+нас)\s+(?:будет\s+)?встреч\w*\b",
            normalized,
        )) and not normalized.endswith("?")

        # Resolve common list/edit phrases deterministically before trusting an LLM label.
        # In particular, requests for active work must not become a fuzzy keyword search.
        if active_query:
            parsed["action"] = "list"
            if re.search(r"\b(?:встреч\w*)\b", normalized):
                parsed["kind"] = "meeting"
            elif re.search(r"\b(?:напоминани\w*|напоминалк\w*)\b", normalized):
                parsed["kind"] = "reminder"
            elif re.search(r"\bзадач\w*\b", normalized):
                parsed["kind"] = "task"
            else:
                parsed["kind"] = "any"
            if period_match:
                phrase = period_match.group(1)
                if "послезавтра" in phrase:
                    parsed["period"] = "day_after_tomorrow"
                elif "завтра" in phrase:
                    parsed["period"] = "tomorrow"
                elif "сегодня" in phrase:
                    parsed["period"] = "today"
                elif "следующей неделе" in phrase:
                    parsed["period"] = "next_week"
                else:
                    parsed["period"] = "week"
            else:
                active_date = self._extract_date_phrase(user_text)
                written_months = "|".join(re.escape(m) for m in MONTHS_RU_PARSE)
                active_date_pattern = r"\b(?:" + "|".join(re.escape(w) for w in WEEKDAYS) + r"|\d{1,2}[.\-/]\d{1,2}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:" + written_months + r"))\b"
                if active_date and re.search(active_date_pattern, normalized):
                    parsed["period"] = "specific_date"
                    parsed["date_text"] = active_date
                else:
                    parsed["period"] = "all"
            parsed["title"] = ""
        elif asks_for_period and not is_create_utterance:
            parsed["action"] = "list"
            period = period_match.group(1)
            if "послезавтра" in period:
                parsed["period"] = "day_after_tomorrow"
            elif "завтра" in period:
                parsed["period"] = "tomorrow"
            elif "сегодня" in period:
                parsed["period"] = "today"
            elif "следующей неделе" in period:
                parsed["period"] = "next_week"
            else:
                parsed["period"] = "week"
            parsed["title"] = ""
            parsed["date_text"] = ""
        elif not is_create_utterance and list_lead:
            # Queries naming a weekday or a calendar date should list exactly that day,
            # not accidentally use an LLM-selected rolling period.
            date_for_list = self._extract_date_phrase(user_text)
            written_months = "|".join(re.escape(m) for m in MONTHS_RU_PARSE)
            date_pattern = r"\b(?:" + "|".join(re.escape(w) for w in WEEKDAYS) + r"|\d{1,2}[.\-/]\d{1,2}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:" + written_months + r"))\b"
            if date_for_list and re.search(date_pattern, normalized):
                parsed["action"] = "list"
                parsed["period"] = "specific_date"
                parsed["date_text"] = date_for_list
                parsed["title"] = ""

        explicit_rename = bool(re.search(
            r"\b(?:переименуй|переименовать|измени\s+(?:название|имя)|изменить\s+(?:название|имя)|поменяй\s+(?:название|имя)|поменять\s+(?:название|имя)|замени\s+(?:название|имя))\b",
            normalized,
        ))
        generic_change = bool(re.match(
            r"^(?:пожалуйста\s+)?(?:измени|изменить|поменяй|поменять|отредактируй|редактируй|обнови|обновить)\s+",
            normalized,
        ))
        has_explicit_schedule = bool(self._extract_date_phrase(user_text) or self._extract_explicit_time(user_text) is not None)
        explicit_reschedule = bool(re.search(
            r"\b(?:перенеси|перенести|перенос|сдвинь|сдвинуть|измени\s+(?:дату|время|срок)|изменить\s+(?:дату|время|срок)|поменяй\s+(?:дату|время|срок)|поменять\s+(?:дату|время|срок)|отложи|отложить)\b",
            normalized,
        ))
        if not is_create_utterance and not natural_task_create and not natural_meeting_create:
            if explicit_rename or (generic_change and not has_explicit_schedule and re.search(r"\b(?:на|вместо)\b", normalized)):
                parsed["action"] = "rename"
            elif explicit_reschedule or (generic_change and has_explicit_schedule):
                parsed["action"] = "reschedule"
            elif generic_change:
                parsed["action"] = "edit_unclear"

        action = parsed.get("action", "unknown")
        kind = parsed.get("kind", "any")
        if kind not in {"task", "reminder", "meeting", "any"}:
            kind = "any"
        title = str(parsed.get("title") or "").strip()
        # Never take dates/times from the LLM on creation/edit: extract them from
        # the user's utterance, otherwise ask a follow-up instead of using a guess.
        source_date = self._extract_date_phrase(user_text)
        date_text = source_date if source_date else ""

        meeting_create = bool(re.match(
            r"^(?:пожалуйста\s+)?(?:добавь|добавить|создай|создать|поставь|поставить|запланируй|запланировать|назначь|назначить)\s+(?:(?:мне|нам)\s+)?(?:новую\s+)?встреч\w*\b",
            normalized,
        )) or natural_meeting_create
        if is_create_utterance or natural_task_create or natural_meeting_create:
            action = "create"
            if meeting_create or natural_meeting_create:
                kind = "meeting"
            elif re.search(r"\b(?:напомни|напомнить|напоминани\w*|напоминалк\w*)\b", normalized):
                kind = "reminder"
            elif re.search(r"\b(?:задач\w*|дело)\b", normalized) or natural_task_create:
                kind = "task"
            extracted_title = self._extract_title_from_request(user_text)
            if extracted_title:
                title = extracted_title
        if action == "reschedule":
            date_text = source_date
        due_at = self._parse_date(date_text)

        if action == "list":
            all_requested = bool(re.search(
                r"\b(?:все\s+(?:записи|задачи|встречи|события|напоминания)?|всё|всех\s+(?:задач|встреч|событий|напоминаний))\b",
                normalized,
            ))
            if all_requested and not period_match and parsed.get("period") != "specific_date":
                parsed["period"] = "all"
            elif parsed.get("period") != "specific_date" and not period_match and not active_query:
                # A model may fill `week` by default; that is not user consent to a period.
                parsed["period"] = ""

        if action == "clarify":
            return str(parsed.get("clarification") or "Уточните, пожалуйста, дату или время.")
        if action == "unknown":
            return "Не понял запрос. Попробуйте: «добавь задачу…», «что у меня завтра?», «найди задачу…»"
        if action == "create":
            if not title:
                return "Не удалось определить название. Что именно нужно записать?"
            if kind == "any":
                explicit_time_for_kind = self._extract_explicit_time(user_text)
                self.pending_clarification = {
                    "type": "create_kind", "kind": "any", "title": title,
                    "date_text": date_text, "time_value": explicit_time_for_kind,
                }
                return "Что создать: задачу, напоминание или встречу?"

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
        if action == "edit_unclear":
            row, candidates, explicit_id = self._resolve_mutation_target(user_text, title, kind)
            if explicit_id is not None and row is None:
                return f"Не нашёл запись с ID {explicit_id}. Ничего не изменено."
            if row is None and len(candidates) > 1:
                choices = "; ".join(f"{r['title']} (ID {r['id']}, {self._format_due(r['due_at'])})" for r in candidates[:5])
                return f"Нашёл несколько похожих записей: {choices}. Уточните название или ID."
            if row is None:
                return "Не нашёл запись для изменения. Укажите её точное название или ID."
            self.pending_clarification = {
                "type": "edit_field", "entry_id": row["id"],
                "title": row["title"], "kind": row["kind"],
            }
            return f"Что изменить у записи «{row['title']}»: название или срок (дату и время)?"
        if action == "list":
            period = parsed.get("period")
            if not period:
                return "За какой период показать записи: сегодня, завтра, на неделю или все записи?"
            if period == "all":
                rows = self.repo.all(100, include_done=not active_query)
                if kind != "any":
                    rows = [r for r in rows if r["kind"] == kind]
            elif period == "specific_date":
                date_value = self._parse_date(parsed.get("date_text", ""))
                if not date_value:
                    return "Какую дату проверить? Укажите её словами или в формате ДД.ММ.ГГГГ."
                date_obj = datetime.fromisoformat(date_value).date()
                start_dt = datetime.combine(date_obj, datetime.min.time())
                rows = self.repo.between(start_dt.isoformat(timespec="minutes"), (start_dt + timedelta(days=1)).isoformat(timespec="minutes"), kind, include_done=not active_query)
            else:
                if period not in {"today", "tomorrow", "day_after_tomorrow", "week", "next_week"}:
                    return "За какой период показать записи: сегодня, завтра, на неделю или все записи?"
                start, end = self._period_bounds(period)
                rows = self.repo.between(start, end, kind, include_done=not active_query)
            if not rows:
                return "Активных записей не найдено." if active_query else "За указанный период записей не найдено."
            prefix = "Активные записи:\n" if active_query else "Найдено:\n"
            return prefix + "\n".join(self._format_entry(r) for r in rows[:20])
        if action == "search":
            query = title or user_text
            rows = self.repo.find(query, kind)
            # If lexical matching is weak or empty, try local multilingual embeddings.
            strong_lexical_match = any(
                query.casefold().strip() in row["title"].casefold()
                or row["title"].casefold() in query.casefold().strip()
                for row in rows if len(query.strip()) >= 4
            )
            if len(rows) < 3 or not strong_lexical_match:
                candidates = [r for r in self.repo.all(1000) if kind in {"any", r["kind"]}]
                try:
                    semantic_rows = self.semantic_search.rank(user_text, candidates, limit=10)
                    known = {r["id"] for r in rows}
                    semantic_additions = [r for r in semantic_rows if r["id"] not in known]
                    if not strong_lexical_match:
                        rows = semantic_additions + rows
                    else:
                        rows.extend(semantic_additions)
                except Exception:
                    # Semantic model is optional until its first successful local download.
                    pass
            if not rows:
                return f"Не нашёл запись по запросу «{query}»."
            return "Нашёл:\n" + "\n".join(self._format_entry(r) for r in rows[:10])
        if action in {"delete", "reschedule", "complete", "rename"}:
            row, matches, explicit_id = self._resolve_mutation_target(user_text, title, kind)
            if explicit_id is not None and row is None:
                return f"Не нашёл запись с ID {explicit_id}. Ничего не изменено."
            if row is None and len(matches) > 1:
                choices = "; ".join(f"{r['title']} (ID {r['id']}, {self._format_due(r['due_at'])})" for r in matches[:5])
                return f"Нашёл несколько похожих записей: {choices}. Уточните название, дату или ID."
            if row is None:
                return "Не нашёл запись. Назовите её точнее или укажите ID."
            if action == "rename":
                new_title = self._extract_new_title(user_text, row)
                if not new_title:
                    self.pending_clarification = {
                        "type": "rename_title", "entry_id": row["id"],
                        "title": row["title"], "kind": row["kind"],
                    }
                    return f"На какое новое название изменить запись «{row['title']}»?"
                if len(new_title) > 200:
                    return "Новое название слишком длинное. Укажите не более 200 символов."
                updated = self.repo.update_title(row["id"], new_title)
                if not updated:
                    return "Не удалось изменить название: запись не найдена."
                return "Название изменено: " + self._format_entry(self.repo.get(row["id"]))
            if action == "delete":
                self.repo.delete(row["id"])
                return f"Удалено: {KIND_RU[row['kind']]} «{row['title']}» (ID {row['id']})."
            if action == "complete":
                self.repo.complete(row["id"])
                return f"Отмечено выполненным: «{row['title']}»."
            explicit_time = self._extract_explicit_time(user_text)
            # A time-only reschedule preserves the record's existing calendar date.
            if not date_text and explicit_time is not None and row.get("due_at"):
                try:
                    existing_dt = datetime.fromisoformat(row["due_at"])
                except (TypeError, ValueError):
                    existing_dt = None
                if existing_dt is not None:
                    due_at = existing_dt.replace(hour=explicit_time[0], minute=explicit_time[1], second=0, microsecond=0).isoformat(timespec="minutes")
                    if not self.repo.update_due(row["id"], due_at):
                        return "Не удалось изменить срок записи."
                    return "Перенесено: " + self._format_entry(self.repo.get(row["id"]))
            if not date_text or due_at is None:
                self.pending_clarification = {
                    "type": "reschedule_date", "entry_id": row["id"],
                    "title": row["title"], "kind": row["kind"],
                    "date_text": "", "time_value": explicit_time,
                }
                return f"На какую дату перенести «{row['title']}»?"
            if explicit_time is None:
                self.pending_clarification = {
                    "type": "reschedule_time", "entry_id": row["id"],
                    "title": row["title"], "kind": row["kind"],
                    "date_text": date_text, "time_value": None,
                }
                return f"Дата: {self._format_date_only(due_at)}. На какое точное время перенести «{row['title']}»?"
            parsed_day = datetime.fromisoformat(due_at)
            due_at = parsed_day.replace(hour=explicit_time[0], minute=explicit_time[1], second=0, microsecond=0).isoformat(timespec="minutes")
            self.repo.update_due(row["id"], due_at)
            return "Перенесено: " + self._format_entry(self.repo.get(row["id"]))
        return "Не удалось выполнить действие. Попробуйте сформулировать запрос иначе."
